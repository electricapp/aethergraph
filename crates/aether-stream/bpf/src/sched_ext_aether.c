// K4.1 sched_ext BPF scheduler — SQPOLL never migrated; everything else FIFO.
//
// Built with feature `sched_ext_bpf` (clang -target bpf -g). Freestanding:
// the kernel types it touches are declared locally with
// preserve_access_index, so libbpf relocates them against the running
// kernel's BTF and no vmlinux.h is needed at build time.
//
// Policy mirror of SchedExtPolicy:
//   - Tasks tagged SQPOLL (task_role == 1) → the published sqpoll CPU, then
//     that CPU's local DSQ
//   - Everything else → the global DSQ, which the core consumes on its own,
//     so no ops.dispatch is needed
//
// Every enqueue inserts the task into a DSQ: a task left out of every DSQ
// never runs again until the watchdog ejects the scheduler.
//
// TODO(HARDWARE): load on a rooted VM; verify SQPOLL run-queue continuity.

#define SEC(NAME) __attribute__((section(NAME), used))
#define __uint(name, val) int (*name)[val]
#define __ksym __attribute__((section(".ksyms")))
#define __weak __attribute__((weak))

typedef unsigned int u32;
typedef unsigned long long u64;

// Kernel constants (include/linux/sched/ext.h).
#define SCX_DSQ_FLAG_BUILTIN (1ULL << 63)
#define SCX_DSQ_GLOBAL (SCX_DSQ_FLAG_BUILTIN | 1)
#define SCX_DSQ_LOCAL (SCX_DSQ_FLAG_BUILTIN | 2)
#define SCX_SLICE_DFL (20ULL * 1000 * 1000)

#define ROLE_SQPOLL 1

struct task_struct {
	int pid;
} __attribute__((preserve_access_index));

// Enqueue kfunc: renamed from scx_bpf_dispatch in 6.13. Both are weak so
// the object loads on either side of the rename; the verifier prunes the
// branch whose symbol did not resolve.
extern void scx_bpf_dsq_insert(struct task_struct *p, u64 dsq_id, u64 slice,
			       u64 enq_flags) __ksym __weak;
extern void scx_bpf_dispatch(struct task_struct *p, u64 dsq_id, u64 slice,
			     u64 enq_flags) __ksym __weak;

static void *(*bpf_map_lookup_elem)(void *map, const void *key) = (void *)1;

struct {
	__uint(type, 1); /* BPF_MAP_TYPE_ARRAY */
	__uint(max_entries, 1);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(u32));
} nic_numa SEC(".maps");

struct {
	__uint(type, 1);
	__uint(max_entries, 256);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(u32));
} nic_numa_cpus SEC(".maps");

/* SQPOLL CPU + 1; 0 means unpublished (array slots start zeroed and CPU 0
 * is a real CPU). */
struct {
	__uint(type, 1);
	__uint(max_entries, 1);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(u32));
} sqpoll_cpu SEC(".maps");

/* Task role: 0=other, 1=sqpoll, 2=gather — set from userspace, keyed by pid. */
struct {
	__uint(type, 2); /* BPF_MAP_TYPE_HASH */
	__uint(max_entries, 4096);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(u32));
} task_role SEC(".maps");

static __attribute__((always_inline)) u32 role_of(struct task_struct *p)
{
	u32 pid = (u32)p->pid;
	u32 *role = bpf_map_lookup_elem(&task_role, &pid);
	return role ? *role : 0;
}

static __attribute__((always_inline)) void insert(struct task_struct *p, u64 dsq,
						   u64 enq_flags)
{
	if (scx_bpf_dsq_insert)
		scx_bpf_dsq_insert(p, dsq, SCX_SLICE_DFL, enq_flags);
	else
		scx_bpf_dispatch(p, dsq, SCX_SLICE_DFL, enq_flags);
}

// struct_ops programs take their arguments as a u64 array.

SEC("struct_ops/aether_select_cpu")
int aether_select_cpu(u64 *ctx)
{
	struct task_struct *p = (struct task_struct *)ctx[0];
	int prev_cpu = (int)ctx[1];
	u32 key = 0;
	u32 *sticky;

	if (role_of(p) != ROLE_SQPOLL)
		return prev_cpu;
	sticky = bpf_map_lookup_elem(&sqpoll_cpu, &key);
	if (sticky && *sticky != 0)
		return (int)(*sticky - 1);
	return prev_cpu;
}

SEC("struct_ops/aether_enqueue")
int aether_enqueue(u64 *ctx)
{
	struct task_struct *p = (struct task_struct *)ctx[0];
	u64 enq_flags = ctx[1];

	/* SQPOLL stays on the CPU select_cpu chose; never stolen. */
	insert(p, role_of(p) == ROLE_SQPOLL ? SCX_DSQ_LOCAL : SCX_DSQ_GLOBAL, enq_flags);
	return 0;
}

/* Local subset of the kernel's struct sched_ext_ops: libbpf matches members
 * by name against kernel BTF and leaves the rest zeroed. */
struct sched_ext_ops {
	int (*select_cpu)(struct task_struct *p, int prev_cpu, u64 wake_flags);
	void (*enqueue)(struct task_struct *p, u64 enq_flags);
	char name[128];
};

SEC(".struct_ops.link")
struct sched_ext_ops aether_ops = {
	.select_cpu = (void *)aether_select_cpu,
	.enqueue = (void *)aether_enqueue,
	.name = "aether",
};

char _license[] SEC("license") = "Dual BSD/GPL";
