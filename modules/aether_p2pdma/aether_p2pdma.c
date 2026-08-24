/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * aether_p2pdma — K3.1 out-of-tree peer-DMA validator (KERNELS.md).
 *
 * Build:
 *   make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
 * Load:
 *   sudo insmod aether_p2pdma.ko
 * Device node:
 *   /dev/aether_p2pdma
 */

#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/pci.h>
#include <linux/dma-buf.h>
#include <linux/slab.h>
#include <linux/version.h>
#include <linux/iommu.h>
#include <linux/string.h>

#include "aether_p2pdma.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("AetherGraph");
MODULE_DESCRIPTION("AetherGraph K3.1 p2pdma path validator");
MODULE_VERSION("0.1");

static int aether_p2pdma_parse_bdf(const char *s, struct pci_dev **out)
{
	unsigned int domain = 0, bus = 0, slot = 0, func = 0;
	int n;

	if (!s || !s[0])
		return -EINVAL;

	n = sscanf(s, "%x:%x:%x.%x", &domain, &bus, &slot, &func);
	if (n != 4) {
		domain = 0;
		n = sscanf(s, "%x:%x.%x", &bus, &slot, &func);
		if (n != 3)
			return -EINVAL;
	}

	*out = pci_get_domain_bus_and_slot(domain, bus, PCI_DEVFN(slot, func));
	return *out ? 0 : -ENODEV;
}

static long aether_p2pdma_ioctl(struct file *file, unsigned int cmd,
			       unsigned long arg)
{
	struct aether_p2pdma_ioctl msg;
	struct pci_dev *producer = NULL, *consumer = NULL;
	struct dma_buf *dmabuf = NULL;
	struct dma_buf_attachment *attach = NULL;
	struct sg_table *sgt = NULL;
	int err = 0;
	int distance = -1;

	(void)file;

	if (cmd != AETHER_P2PDMA_VALIDATE)
		return -ENOTTY;

	if (copy_from_user(&msg, (void __user *)arg, sizeof(msg)))
		return -EFAULT;

	msg.req.producer_bdf[sizeof(msg.req.producer_bdf) - 1] = '\0';
	msg.req.consumer_bdf[sizeof(msg.req.consumer_bdf) - 1] = '\0';
	memset(&msg.resp, 0, sizeof(msg.resp));

	err = aether_p2pdma_parse_bdf(msg.req.producer_bdf, &producer);
	if (err)
		goto out;
	err = aether_p2pdma_parse_bdf(msg.req.consumer_bdf, &consumer);
	if (err)
		goto out;

	dmabuf = dma_buf_get(msg.req.dmabuf_fd);
	if (IS_ERR(dmabuf)) {
		err = PTR_ERR(dmabuf);
		dmabuf = NULL;
		goto out;
	}

	attach = dma_buf_attach(dmabuf, &consumer->dev);
	if (IS_ERR(attach)) {
		err = PTR_ERR(attach);
		attach = NULL;
		goto out;
	}

	sgt = dma_buf_map_attachment(attach, DMA_BIDIRECTIONAL);
	if (IS_ERR(sgt)) {
		err = PTR_ERR(sgt);
		sgt = NULL;
		goto out;
	}

#if defined(CONFIG_PCI_P2PDMA) && LINUX_VERSION_CODE >= KERNEL_VERSION(5, 12, 0)
	distance = pci_p2pdma_distance(producer, consumer, true);
#else
	distance = -1;
#endif

	if (distance < 0) {
		msg.resp.status = AETHER_P2PDMA_UNSUPPORTED;
	} else if (msg.req.require_iommu && !iommu_present(&pci_bus_type)) {
		msg.resp.status = AETHER_P2PDMA_NO_IOMMU;
	} else if ((u32)distance > msg.req.maximum_distance) {
		msg.resp.status = AETHER_P2PDMA_TOO_FAR;
		msg.resp.distance = (u32)distance;
	} else {
		msg.resp.status = AETHER_P2PDMA_OK;
		msg.resp.distance = (u32)distance;
		if (sgt->nents && sgt->sgl)
			msg.resp.peer_bus_addr = sg_dma_address(sgt->sgl);
	}

out:
	if (sgt && !IS_ERR_OR_NULL(sgt))
		dma_buf_unmap_attachment(attach, sgt, DMA_BIDIRECTIONAL);
	if (attach && !IS_ERR_OR_NULL(attach))
		dma_buf_detach(dmabuf, attach);
	if (dmabuf)
		dma_buf_put(dmabuf);
	if (consumer)
		pci_dev_put(consumer);
	if (producer)
		pci_dev_put(producer);

	if (err)
		return err;

	if (copy_to_user((void __user *)arg, &msg, sizeof(msg)))
		return -EFAULT;
	return 0;
}

static const struct file_operations aether_p2pdma_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = aether_p2pdma_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl = aether_p2pdma_ioctl,
#endif
};

static struct miscdevice aether_p2pdma_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "aether_p2pdma",
	.fops = &aether_p2pdma_fops,
	.mode = 0600,
};

static int __init aether_p2pdma_init(void)
{
	return misc_register(&aether_p2pdma_dev);
}

static void __exit aether_p2pdma_exit(void)
{
	misc_deregister(&aether_p2pdma_dev);
}

module_init(aether_p2pdma_init);
module_exit(aether_p2pdma_exit);
