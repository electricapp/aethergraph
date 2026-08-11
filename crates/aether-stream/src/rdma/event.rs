//! Event-driven completion waiting — the low-CPU alternative to busy
//! polling a CQ.
//!
//! A CQ created on a [`CompletionChannel`] delivers an event through the
//! channel's fd each time a completion arrives while the CQ is armed.
//! The consumer loop is: [`CompletionChannel::arm`] → drain the CQ →
//! [`CompletionChannel::wait`] → drain again. Arming *before* the drain
//! closes the race where a completion lands between drain and wait —
//! such a completion fires the armed event, so the wait returns
//! immediately instead of sleeping through it.
//!
//! Busy-polling stays the right choice under sustained load; the channel
//! is for the idle side of an adaptive consumer, where burning a core to
//! poll an empty CQ is pure waste.

use super::context::{RdmaContext, RegisteredCq, create_cq_on_channel};
use super::ffi::*;
use std::io;
use std::time::{Duration, Instant};

/// Owns an `ibv_comp_channel`. Create the channel first, then CQs on it
/// via [`Self::create_cq`]; drop those CQs before the channel, and the
/// channel before its context.
pub struct CompletionChannel {
    channel: *mut IbvCompChannel,
}

// SAFETY: ibverbs completion channels are thread-safe after creation.
unsafe impl Send for CompletionChannel {}
// SAFETY: see Send impl above.
unsafe impl Sync for CompletionChannel {}

impl CompletionChannel {
    pub fn create(ctx: &RdmaContext) -> io::Result<Self> {
        // SAFETY: `ctx.context_ptr()` is an open device context.
        let channel = unsafe { ibv_create_comp_channel(ctx.context_ptr()) };
        if channel.is_null() {
            return Err(io::Error::other("ibv_create_comp_channel failed"));
        }
        Ok(Self { channel })
    }

    /// A CQ whose completions raise events on this channel.
    pub fn create_cq(&self, ctx: &RdmaContext, cq_size: i32) -> io::Result<RegisteredCq> {
        create_cq_on_channel(ctx, cq_size, self.channel)
    }

    /// The channel's file descriptor, for integration into an external
    /// event loop (epoll/poll). [`Self::wait`] polls it internally.
    pub fn fd(&self) -> i32 {
        // SAFETY: `self.channel` is alive; `fd` is public ABI.
        unsafe { (*self.channel).fd }
    }

    /// Arm `cq`: the next completion added to it raises one event on
    /// this channel. One-shot — re-arm after every wakeup.
    ///
    /// With `solicited_only`, only completions whose sender set
    /// `IBV_SEND_SOLICITED` (and errors) raise the event.
    pub fn arm(&self, cq: &RegisteredCq, solicited_only: bool) -> io::Result<()> {
        // SAFETY: `cq` is a live CQ created on this channel.
        let ret = unsafe { ibv_req_notify_cq(cq.as_ptr(), i32::from(solicited_only)) };
        if ret != 0 {
            return Err(io::Error::other(format!("ibv_req_notify_cq failed: {ret}")));
        }
        Ok(())
    }

    /// Block until an armed CQ raises an event or `timeout` passes.
    /// Returns `true` when an event arrived (already acknowledged) —
    /// drain the CQ and re-arm; `false` on timeout.
    pub fn wait(&self, timeout: Option<Duration>) -> io::Result<bool> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut pollfd = libc::pollfd {
            fd: self.fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            let wait_ms: i32 = match deadline {
                None => -1,
                Some(d) => {
                    let left = d.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Ok(false);
                    }
                    left.as_millis().min(i32::MAX as u128) as i32
                }
            };
            // SAFETY: `pollfd` is a valid single-entry array.
            let ret = unsafe { libc::poll(&mut pollfd, 1, wait_ms) };
            match ret {
                0 => return Ok(false),
                r if r < 0 => {
                    let err = io::Error::last_os_error();
                    if err.raw_os_error() == Some(libc::EINTR) {
                        continue;
                    }
                    return Err(err);
                }
                _ => break,
            }
        }

        // The fd is readable: collect exactly one event and acknowledge
        // it so the CQ's event counter stays balanced for destroy.
        let mut cq: *mut IbvCq = std::ptr::null_mut();
        let mut cq_context: *mut libc::c_void = std::ptr::null_mut();
        // SAFETY: `self.channel` is alive; both out-pointers are valid.
        let ret = unsafe { ibv_get_cq_event(self.channel, &mut cq, &mut cq_context) };
        if ret != 0 {
            return Err(io::Error::other(format!("ibv_get_cq_event failed: {ret}")));
        }
        // SAFETY: `cq` came from the event we just collected.
        unsafe { ibv_ack_cq_events(cq, 1) };
        Ok(true)
    }
}

impl Drop for CompletionChannel {
    fn drop(&mut self) {
        // SAFETY: `self.channel` was created by ibv_create_comp_channel
        // and every CQ on it has been dropped per the struct contract.
        let ret = unsafe { ibv_destroy_comp_channel(self.channel) };
        if ret != 0 {
            tracing::warn!(ret, "ibv_destroy_comp_channel failed");
        }
    }
}
