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
/// * `core_id` - CPU core to pin this thread to
pub fn ingest_loop(
    socket: &mut XdpSocket,
    umem: &Arc<Umem>,
    tx: &Sender<InboundFrame>,
    config: &IngestConfig,
) {
    loop {
        // 1. Drain completion ring → return frames to UMEM
        drain_completions(socket, umem);

        // 2. Refill fill ring with free frames
        refill_fill_ring(socket, umem, config.fill_batch_size);

        // 3. Poll RX ring
        let available = socket.rx.available();
        if available == 0 {
            std::hint::spin_loop();
            continue;
        }

        let batch = available.min(config.rx_batch_size);
        for i in 0..batch {
            // SAFETY: i < available, so peek is valid
            let desc: RxTxDesc = unsafe { socket.rx.peek(i) };

            // Convert UMEM addr to frame index
            let frame_idx = (desc.addr / umem.frame_size() as u64) as usize;

            let frame = InboundFrame {
                umem_idx: frame_idx,
                len: desc.len,
                data: umem.frame_ptr(frame_idx),
            };

            // If channel is full/disconnected, release frame back
            if tx.send(frame).is_err() {
                umem.release_frame(frame_idx);
                return; // receiver gone, shut down
            }
        }

        // SAFETY: we consumed `batch` entries
        unsafe {
            socket.rx.advance_consumer(batch);
        }
    }
}

/// Drain the completion ring and return frames to the UMEM pool.
fn drain_completions(socket: &mut XdpSocket, umem: &Umem) {
    let available = socket.completion.available();
    for i in 0..available {
        // SAFETY: i < available
        let addr: u64 = unsafe { socket.completion.peek(i) };
        let frame_idx = (addr / umem.frame_size() as u64) as usize;
        umem.release_frame(frame_idx);
    }
    if available > 0 {
        // SAFETY: we consumed `available` entries
        unsafe {
            socket.completion.advance_consumer(available);
        }
    }
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
}

/// Spawn ingestion threads — one per socket, each pinned to a core.
///
/// Returns the crossbeam receiver for inbound frames.
pub fn spawn_ingest_threads(
    mut sockets: Vec<XdpSocket>,
    umem: Arc<Umem>,
    core_ids: &[usize],
    config: IngestConfig,
) -> crossbeam_channel::Receiver<InboundFrame> {
    let (tx, rx) = crossbeam_channel::bounded(sockets.len() * 1024);

    for (i, mut socket) in sockets.drain(..).enumerate() {
        let umem = umem.clone();
        let tx = tx.clone();
        let config = config.clone();
        let core_id = core_ids.get(i).copied();

        std::thread::Builder::new()
            .name(format!("ingest-q{}", i))
            .spawn(move || {
                // Pin to core if specified
                if let Some(id) = core_id {
                    let core = core_affinity::CoreId { id };
                    core_affinity::set_for_current(core);
                    tracing::info!(core = id, queue = i, "Ingestion thread pinned to core");
                }

                ingest_loop(&mut socket, &umem, &tx, &config);
                tracing::info!(queue = i, "Ingestion thread exiting");
            })
            .expect("failed to spawn ingestion thread");
    }

    rx
}
