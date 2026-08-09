//! Busy-poll ingestion loop — one thread per NIC queue, pinned to a core.
//!
//! Each thread:
//! 1. Busy-polls the RX ring (spins on empty)
//! 2. Drains the COMPLETION ring → returns frames to UMEM free list
//! 3. Refills the FILL ring with free frames
//! 4. Sends received frames via crossbeam channel

use crate::umem::Umem;
use crate::xdp::rings::RxTxDesc;
use crate::xdp::socket::XdpSocket;
use crossbeam_channel::Sender;
use std::sync::Arc;

/// An inbound frame received from the NIC.
#[derive(Debug)]
pub struct InboundFrame {
    /// UMEM frame index (for releasing back to pool after processing).
    pub umem_idx: usize,
    /// Length of valid data.
    pub len: u32,
    /// Raw pointer to frame data in UMEM.
    /// Valid until `umem.release_frame(umem_idx)` is called.
    pub data: *const u8,
}

// SAFETY: InboundFrame data pointer is valid for its UMEM frame lifetime
unsafe impl Send for InboundFrame {}

/// Configuration for the ingestion loop.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IngestConfig {
    /// Maximum number of frames to batch from the RX ring per iteration.
    pub rx_batch_size: u32,
    /// Maximum number of frames to refill into the FILL ring per iteration.
    pub fill_batch_size: u32,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            rx_batch_size: 64,
            fill_batch_size: 64,
        }
    }
}

/// Run the ingestion loop for a single NIC queue.
///
/// This function does not return — it busy-polls in a loop until the
/// channel is disconnected (receiver dropped).
///
/// # Arguments
/// * `socket` - AF_XDP socket bound to a NIC queue
/// * `umem` - Shared UMEM frame pool
/// * `tx` - Channel for sending received frames to processing threads
/// * `config` - Batching configuration
///
/// Core pinning is done by the caller ([`spawn_ingest_threads`]) before this
/// loop starts, not here.
pub fn ingest_loop(
    socket: &mut XdpSocket,
    umem: &Arc<Umem>,
    tx: &Sender<Vec<InboundFrame>>,
    config: &IngestConfig,
) {
    // Frame size is a power of two (asserted by Umem::new); shift/mask
    // replace the u64 divide + modulo the loop would otherwise pay per
    // descriptor.
    let frame_shift = umem.frame_size().trailing_zeros();
    let frame_mask = (umem.frame_size() - 1) as u64;
    let mut completed_scratch: Vec<usize> = Vec::with_capacity(config.fill_batch_size as usize);

    loop {
        // 1. Drain completion ring → return frames to UMEM
        drain_completions(socket, umem, &mut completed_scratch, frame_shift);

        // 2. Refill fill ring with free frames
        refill_fill_ring(socket, umem, config.fill_batch_size);

        // 3. Poll RX ring
        let available = socket.rx.available();
        if available == 0 {
            std::hint::spin_loop();
            continue;
        }

        let batch = available.min(config.rx_batch_size);
        // One channel operation per RX batch, not per frame: at line rate a
        // per-packet MPMC send (atomic CAS + possible futex wake each)
        // becomes the throughput ceiling before the parser does.
        let mut frames: Vec<InboundFrame> = Vec::with_capacity(batch as usize);
        for i in 0..batch {
            // SAFETY: i < available, so peek is valid
            let desc: RxTxDesc = unsafe { socket.rx.peek(i) };

            // Convert UMEM addr to frame index. The kernel supplies `desc.addr`;
            // a buggy/hostile driver could hand back an address outside the
            // UMEM, so validate before turning it into a pointer (the
            // `frame_ptr` bound is only a debug_assert and would be UB in
            // release). Drop the descriptor on violation; it's not a UMEM frame
            // we own, so we must not release it back to the pool.
            let frame_idx = (desc.addr >> frame_shift) as usize;
            let end = desc.addr.checked_add(desc.len as u64);
            if end.is_none_or(|e| e > umem.total_size() as u64) || frame_idx >= umem.frame_count() {
                tracing::warn!(
                    addr = desc.addr,
                    len = desc.len,
                    frame_idx,
                    "RX descriptor [addr, addr+len) out of UMEM bounds; dropping frame"
                );
                continue;
            }

            // `desc.addr` points at the packet data itself, which the kernel
            // places at an offset inside the chunk (XDP headroom). Keep that
            // offset — the chunk base holds stale bytes, not the packet.
            let offset_in_frame = (desc.addr & frame_mask) as usize;
            frames.push(InboundFrame {
                umem_idx: frame_idx,
                len: desc.len,
                // SAFETY: [desc.addr, desc.addr + desc.len) is bounds-checked
                // against the UMEM above; frame stride equals frame_size
                // (enforced by Umem::new), so base + frame_idx * frame_size
                // + offset_in_frame == base + desc.addr.
                data: unsafe { umem.frame_ptr(frame_idx).add(offset_in_frame) },
            });
        }

        // SAFETY: we consumed `batch` entries
        unsafe {
            socket.rx.advance_consumer(batch);
        }

        if frames.is_empty() {
            continue;
        }

        // Bounded channel: `send` BLOCKS when the queue is full — that is
        // the intended backpressure (this pinned busy-poll thread parks
        // until a consumer drains). `Err` only means the receiver
        // disconnected: release the in-hand frames and shut down. Batches
        // already inside the channel are dropped at shutdown.
        if let Err(returned) = tx.send(frames) {
            completed_scratch.clear();
            completed_scratch.extend(returned.0.iter().map(|f| f.umem_idx));
            umem.release_frames(&completed_scratch);
            return; // receiver gone, shut down
        }
    }
}

/// Drain the completion ring and return frames to the UMEM pool in one
/// batched free-list splice (one CAS per drain instead of one per frame).
fn drain_completions(
    socket: &mut XdpSocket,
    umem: &Umem,
    scratch: &mut Vec<usize>,
    frame_shift: u32,
) {
    let available = socket.completion.available();
    if available == 0 {
        return;
    }
    scratch.clear();
    for i in 0..available {
        // SAFETY: i < available
        let addr: u64 = unsafe { socket.completion.peek(i) };
        let frame_idx = (addr >> frame_shift) as usize;
        // Guard the kernel-supplied completion addr before releasing: a
        // bad index would corrupt the UMEM free list.
        if addr >= umem.total_size() as u64 || frame_idx >= umem.frame_count() {
            tracing::warn!(
                addr,
                frame_idx,
                "completion addr out of UMEM bounds; skipping release"
            );
            continue;
        }
        scratch.push(frame_idx);
    }
    // SAFETY: we consumed `available` entries
    unsafe {
        socket.completion.advance_consumer(available);
    }
    umem.release_frames(scratch);
}

/// Refill the fill ring with free frames from the UMEM pool.
fn refill_fill_ring(socket: &mut XdpSocket, umem: &Umem, max_refill: u32) {
    let free_slots = socket.fill.free_slots();
    let to_fill = free_slots.min(max_refill);

    let mut filled = 0u32;
    for i in 0..to_fill {
        let Some(frame_idx) = umem.acquire_frame() else {
            break; // pool exhausted
        };
        let addr = umem.frame_addr(frame_idx);
        // SAFETY: i < free_slots
        unsafe {
            socket.fill.enqueue_at(i, addr);
        }
        filled += 1;
    }

    if filled > 0 {
        // SAFETY: we wrote `filled` entries
        unsafe {
            socket.fill.advance_producer(filled);
        }
    }

    // Zero-copy / busy-poll drivers stop draining the FILL ring once caught up
    // and raise XDP_RING_NEED_WAKEUP. Without a syscall kick the kernel never
    // pulls the descriptors we just published and the RX busy-poll livelocks.
    if socket.fill.needs_wakeup() {
        kick_rx(socket.fd());
    }
}

/// Wake the kernel's RX/FILL processing for a socket whose FILL ring raised
/// `XDP_RING_NEED_WAKEUP`. A non-blocking zero-length `recvfrom` is the
/// documented kick for the RX side; it moves no data, it only nudges the
/// driver. Errors (e.g. `EAGAIN`) are expected and ignored — the next loop
/// iteration re-checks the flag.
fn kick_rx(fd: i32) {
    // SAFETY: `fd` is the live AF_XDP socket fd owned by the XdpSocket; all
    // pointer args are null and the length is 0, so the kernel reads/writes no
    // user buffer. This is the standard AF_XDP wakeup syscall.
    unsafe {
        libc::recvfrom(
            fd,
            std::ptr::null_mut(),
            0,
            libc::MSG_DONTWAIT,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }
}

/// Errors returned by [`spawn_ingest_threads`].
#[derive(Debug)]
pub enum SpawnError {
    /// `std::thread::Builder::spawn` returned an OS error.
    Thread(std::io::Error),
}

impl std::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpawnError::Thread(e) => write!(f, "failed to spawn ingestion thread: {e}"),
        }
    }
}

impl std::error::Error for SpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SpawnError::Thread(e) => Some(e),
        }
    }
}

/// A running ingest pool: the frame receiver plus one join handle per
/// spawned thread.
pub type IngestHandles = (
    crossbeam_channel::Receiver<Vec<InboundFrame>>,
    Vec<std::thread::JoinHandle<()>>,
);

/// Spawn ingestion threads — one per socket, each pinned to a core.
///
/// Returns the crossbeam receiver for inbound frames along with the join
/// handles. On thread spawn failure, already-spawned threads are not cancelled
/// (they keep polling) but the function returns the error so the caller can
/// decide.
pub fn spawn_ingest_threads(
    mut sockets: Vec<XdpSocket>,
    umem: Arc<Umem>,
    core_ids: &[usize],
    config: IngestConfig,
) -> Result<IngestHandles, SpawnError> {
    let (tx, rx) = crossbeam_channel::bounded(sockets.len() * 1024);
    let mut handles = Vec::with_capacity(sockets.len());

    for (i, mut socket) in sockets.drain(..).enumerate() {
        let umem = umem.clone();
        let tx = tx.clone();
        let config = config.clone();
        let core_id = core_ids.get(i).copied();

        let handle = std::thread::Builder::new()
            .name(format!("ingest-q{}", i))
            .spawn(move || {
                if let Some(id) = core_id {
                    let core = core_affinity::CoreId { id };
                    core_affinity::set_for_current(core);
                    tracing::info!(core = id, queue = i, "Ingestion thread pinned to core");
                }

                ingest_loop(&mut socket, &umem, &tx, &config);
                tracing::info!(queue = i, "Ingestion thread exiting");
            })
            .map_err(SpawnError::Thread)?;
        handles.push(handle);
    }

    Ok((rx, handles))
}
