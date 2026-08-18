//! Statically-defined tracepoints (USDT) for zero-overhead production
//! tracing.
//!
//! A USDT probe is a single `nop` in the instruction stream plus a note in
//! the ELF `.note.stapsdt` section describing where that nop is and how to
//! find its arguments. Until something attaches, the cost is the nop —
//! there is no branch, no atomic, no counter. When `bpftrace` or `perf`
//! attaches, the kernel patches the nop into a breakpoint and the probe
//! fires with its arguments readable.
//!
//! That property is what makes these usable on the hot path: unlike
//! [`CounterSet`](super::perf::CounterSet), which brackets a region and is
//! opt-in per measurement, probes can be left in the sampling inner loop
//! permanently and cost nothing when unobserved.
//!
//! Attach to them with, for example:
//!
//! ```text
//! bpftrace -e 'usdt:/path/to/libaethergraph.so:aethergraph:sample_batch_done
//!              { @nodes = hist(arg1); }'
//! ```
//!
//! Probes are ELF-and-x86-64/aarch64 specific. On every other target the
//! macro expands to nothing, so call sites need no `cfg` of their own.

/// Emit a USDT probe with up to four `usize`-typed arguments.
///
/// ```ignore
/// probe!(sample_batch_done, seeds.len(), subgraph.num_nodes());
/// ```
///
/// The provider is always `aethergraph`; the first argument names the
/// probe. Arguments are evaluated only on platforms that emit probes, so
/// keep them cheap — they are reads of values the caller already has, not
/// computations.
///
/// Each argument is held in a register that the probe's note describes as
/// `-8@<reg>`, which is how a tracer resolves `arg1`, `arg2`, … The
/// descriptor is generated from the operands, so the count a call site
/// passes and the count a tracer sees cannot drift apart.
#[macro_export]
macro_rules! probe {
    ($name:ident $(, $arg:expr)* $(,)?) => {
        $crate::__probe_impl!($name $(, $arg)*)
    };
}

/// The platform-specific expansion behind [`probe!`].
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[macro_export]
#[doc(hidden)]
macro_rules! __probe_impl {
    ($name:ident) => {{
        // SAFETY: see `__probe_note` — one `nop` plus assembler directives
        // that emit a note section, touching no memory and clobbering
        // nothing, which `nostack`/`nomem` assert to the optimizer.
        unsafe {
            ::core::arch::asm!(
                $crate::__probe_note!($name, ""),
                options(nostack, nomem, preserves_flags),
            );
        }
    }};
    ($name:ident, $a0:expr) => {{
        let a0: usize = $a0;
        // SAFETY: as above; the operand is only named by the descriptor
        // string, never read or written by the emitted instruction.
        unsafe {
            ::core::arch::asm!(
                $crate::__probe_note!($name, $crate::__probe_arg!(0)),
                in(reg) a0,
                options(nostack, nomem, preserves_flags),
            );
        }
    }};
    ($name:ident, $a0:expr, $a1:expr) => {{
        let a0: usize = $a0;
        let a1: usize = $a1;
        // SAFETY: as above.
        unsafe {
            ::core::arch::asm!(
                $crate::__probe_note!(
                    $name,
                    concat!($crate::__probe_arg!(0), " ", $crate::__probe_arg!(1))
                ),
                in(reg) a0,
                in(reg) a1,
                options(nostack, nomem, preserves_flags),
            );
        }
    }};
    ($name:ident, $a0:expr, $a1:expr, $a2:expr) => {{
        let a0: usize = $a0;
        let a1: usize = $a1;
        let a2: usize = $a2;
        // SAFETY: as above.
        unsafe {
            ::core::arch::asm!(
                $crate::__probe_note!(
                    $name,
                    concat!(
                        $crate::__probe_arg!(0), " ",
                        $crate::__probe_arg!(1), " ",
                        $crate::__probe_arg!(2)
                    )
                ),
                in(reg) a0,
                in(reg) a1,
                in(reg) a2,
                options(nostack, nomem, preserves_flags),
            );
        }
    }};
    ($name:ident, $a0:expr, $a1:expr, $a2:expr, $a3:expr) => {{
        let a0: usize = $a0;
        let a1: usize = $a1;
        let a2: usize = $a2;
        let a3: usize = $a3;
        // SAFETY: as above.
        unsafe {
            ::core::arch::asm!(
                $crate::__probe_note!(
                    $name,
                    concat!(
                        $crate::__probe_arg!(0), " ",
                        $crate::__probe_arg!(1), " ",
                        $crate::__probe_arg!(2), " ",
                        $crate::__probe_arg!(3)
                    )
                ),
                in(reg) a0,
                in(reg) a1,
                in(reg) a2,
                in(reg) a3,
                options(nostack, nomem, preserves_flags),
            );
        }
    }};
}

/// The `.note.stapsdt` entry for one probe, with `$argdesc` as its
/// argument descriptor.
///
/// The descriptor is what a tracer reads to find `arg1`, `arg2`, … An
/// empty one declares a probe that carries no data, however many values
/// the call site passed.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[macro_export]
#[doc(hidden)]
macro_rules! __probe_note {
    ($name:ident, $argdesc:expr) => {
        concat!(
            "990: nop\n",
            ".pushsection .note.stapsdt,\"?\",\"note\"\n",
            ".balign 4\n",
            ".4byte 992f-991f, 994f-993f, 3\n",
            "991: .asciz \"stapsdt\"\n",
            "992: .balign 4\n",
            "993: .8byte 990b\n",       // probe address
            ".8byte _.stapsdt.base\n",  // base for prelink fixups
            ".8byte 0\n",               // semaphore (none)
            ".asciz \"aethergraph\"\n", // provider
            ".asciz \"",
            stringify!($name),
            "\"\n",
            ".asciz \"",
            $argdesc,
            "\"\n", // argument descriptor
            "994: .balign 4\n",
            ".popsection\n",
        )
    };
}

/// One argument's descriptor: 8 bytes, signed, in the register Rust
/// allocated for operand `$n`.
///
/// `{n}` is substituted with that register's name. SystemTap spells x86
/// registers AT&T-style with a leading `%`, which Rust's default Intel
/// syntax does not emit, so it is written literally; aarch64 uses the bare
/// name that substitution already produces.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[macro_export]
#[doc(hidden)]
macro_rules! __probe_arg {
    ($n:literal) => {
        concat!("-8@%{", $n, "}")
    };
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
#[macro_export]
#[doc(hidden)]
macro_rules! __probe_arg {
    ($n:literal) => {
        concat!("-8@{", $n, "}")
    };
}

/// No-op expansion for targets without ELF SDT notes.
#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
#[macro_export]
#[doc(hidden)]
macro_rules! __probe_impl {
    ($name:ident $(, $arg:expr)*) => {{
        // Consume the arguments so call sites type-check identically on
        // every platform and no `unused` warning appears.
        $(
            let _: usize = $arg;
        )*
    }};
}

// Anchor symbol every SDT note references for prelink relocation. One
// weak, hidden definition per shared object; the linker folds the
// duplicates from separate compilation units into one.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
core::arch::global_asm!(
    ".ifndef _.stapsdt.base",
    ".pushsection .stapsdt.base,\"aG\",\"progbits\",.stapsdt.base,comdat",
    ".weak _.stapsdt.base",
    ".hidden _.stapsdt.base",
    "_.stapsdt.base: .space 1",
    ".size _.stapsdt.base, 1",
    ".popsection",
    ".endif",
);

#[cfg(test)]
mod tests {
    /// The macro must compile and run at every supported arity, on every
    /// platform. Firing a probe nobody is attached to is a nop, so from
    /// inside the process the observable contract is only that this
    /// executes — whether the note describes the arguments is a property
    /// of the emitted ELF, checked by reading `.note.stapsdt` out of the
    /// build rather than from here.
    #[test]
    fn probes_fire_without_a_consumer() {
        crate::probe!(test_probe_no_args);
        crate::probe!(test_probe_one_arg, 42usize);
        crate::probe!(test_probe_two_args, 1usize, 2usize);
        crate::probe!(test_probe_three_args, 1usize, 2usize, 3usize);
        crate::probe!(test_probe_four_args, 1usize, 2usize, 3usize, 4usize);
        crate::probe!(test_probe_trailing_comma, 7usize,);
    }

    /// Arguments are ordinary expressions evaluated in place.
    #[test]
    fn probe_arguments_are_expressions() {
        let v = vec![1u32, 2, 3];
        crate::probe!(test_probe_expr, v.len(), v.capacity());
    }
}
