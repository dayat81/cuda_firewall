#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_metrics.h>
#include <rte_bitrate.h>
#include <rte_latencystats.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "rte_bv_classifier.h"
#include "parser.h"
#include "config.h"
#include "misc.h"
#include "stats.h"

#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1
//#define DO_NOT_TRANSMIT_TO_TAP

struct rte_pktmbuf_extmem ext_mem;
struct rte_mempool *mpool_payload;
struct rte_bv_classifier *classifier;
ruleset_t ruleset;
uint16_t tap_port_id, trunk_port_id;

volatile uint8_t running;

unsigned long timestamp;
stats_t *old_port_stats=NULL, *port_stats=NULL;

typedef struct {
    struct rte_ether_addr *tap_macaddr;
} callback_payload_t;

typedef struct {
    struct rte_bv_classifier *classifier;
    struct rte_ether_addr *tap_macaddr;
} trunk_tx_param_t;

void exit_handler(int e) {
    running=0;

    rte_eal_mp_wait_lcore();

    free_ruleset(&ruleset);

    rte_bv_classifier_free(classifier);

    rte_free(old_port_stats);
    rte_free(port_stats);

    rte_eal_cleanup();

    exit(EXIT_SUCCESS);
}

void print_stats(__rte_unused int e) {
    struct timeval t;
    gettimeofday(&t, NULL);

    unsigned long new_ts=1000000*t.tv_sec+t.tv_usec;
    double ts_d=(double) (new_ts-timestamp)/1000000.0;

#define PPS(X, P) (((double) (port_stats[P].X-old_port_stats[P].X))/ts_d)
    printf("[trunk] pkts_in: %.2lfpps pkts_out: %.2lfpps pkts_dropped: %.2lfpps pkts_accepted: %.2lfpps pkts_lookup_hit_miss: %.2lfpps\n",
           PPS(pkts_in, 0), PPS(pkts_out, 0), PPS(pkts_dropped, 0), PPS(pkts_accepted, 0), PPS(pkts_lookup_miss, 0));
    old_port_stats[0]=port_stats[0];

    printf("[fw-tap] pkts_in: %.2lfpps pkts_out: %.2lfpps pkts_dropped: %.2lfpps pkts_accepted: %.2lfpps pkts_lookup_miss: %.2lfpps\n",
           PPS(pkts_in, 1), PPS(pkts_out, 1), PPS(pkts_dropped, 1), PPS(pkts_accepted, 1), PPS(pkts_lookup_miss, 1));
    old_port_stats[1]=port_stats[1];
#undef PPS

    timestamp=new_ts;
}

void tx_callback(struct rte_mbuf **pkts, uint16_t nb_rx, uint8_t *lookup_hit_vec, void **actions_r, void *p_r) {
    uint8_t **actions=(uint8_t **) actions_r;
    uint16_t i=0, j=0;

#ifdef DO_NOT_TRANSMIT_TO_TAP
    for(; i<nb_rx; ++i) {
        if(unlikely(!lookup_hit_vec[i])) {
            ++j;
            port_stats->pkts_lookup_miss++;
            continue;
        }

        if(*(actions[i])==RULE_DROP) {
            continue;
        }

        ++j;
    }

    rte_pktmbuf_free_bulk(pkts, nb_rx);

    port_stats->pkts_in+=nb_rx;
    port_stats->pkts_out+=j;
    port_stats->pkts_accepted+=j;
    port_stats->pkts_dropped+=nb_rx-j;
#else
    callback_payload_t *p=(callback_payload_t *) p_r;
    struct rte_mbuf *bufs_tx[BURST_SIZE];
    for(; i<nb_rx; ++i) {
        if(unlikely(!lookup_hit_vec[i])) {
            bufs_tx[j++]=pkts[i];
            if(p->tap_macaddr)
                rte_memcpy(&(rte_pktmbuf_mtod(pkts[i], struct rte_ether_hdr *)->dst_addr), p->tap_macaddr, 6);
            port_stats->pkts_lookup_miss++;
            continue;
        }

        if(unlikely(*(actions[i])==RULE_DROP)) {
            rte_pktmbuf_free(pkts[i]);
            continue;
        }

        bufs_tx[j++]=pkts[i];
        if(p->tap_macaddr)
            rte_memcpy(&(rte_pktmbuf_mtod(pkts[i], struct rte_ether_hdr *)->dst_addr), p->tap_macaddr, 6);
    }

    const uint16_t nb_tx=rte_eth_tx_burst(tap_port_id, 0, bufs_tx, j);

    port_stats->pkts_in+=nb_rx;
    port_stats->pkts_out+=nb_tx;
    port_stats->pkts_accepted+=j;
    port_stats->pkts_dropped+=nb_rx-j;
#endif
}

int trunk_rx(void *arg) {
    printf("[trunk_rx] launched\n");
    struct rte_bv_classifier *c=(struct rte_bv_classifier *) arg;
    struct rte_mbuf *bufs_rx[BURST_SIZE];

    while(running) {
        const uint16_t nb_rx = rte_eth_rx_burst(trunk_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        rte_bv_classifier_enqueue_burst(c, bufs_rx, nb_rx);
    }

    printf("[trunk_rx] stopped\n");
    return 0;
}

static int trunk_tx(void *p_r) {
    trunk_tx_param_t *p=(trunk_tx_param_t *) p_r;
    printf("[trunk_tx] launched\n");
    callback_payload_t payload= { .tap_macaddr=p->tap_macaddr};

    rte_bv_classifier_poll_lookups(p->classifier, tx_callback, (void *) &payload);
    return 0;
}

static int tap_tx(__rte_unused void *arg) {
    printf("[tap_tx] launched\n");
    struct rte_mbuf *bufs_rx[BURST_SIZE];

    while(running) {
        const uint16_t nb_rx = rte_eth_rx_burst(tap_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        const uint16_t nb_tx = rte_eth_tx_burst(trunk_port_id, 0, bufs_rx, nb_rx);

        if(unlikely(nb_tx<nb_rx)) {
            for(uint16_t b=nb_tx; b<nb_rx; ++b)
                rte_pktmbuf_free(bufs_rx[b]);
        }
    }

    printf("[tap_tx] stopped\n");
    return 0;
}

static uint8_t find_tap_trunk_devs(uint16_t *tap_id, uint16_t *trunk_id) {
    struct rte_eth_dev_info dev_info;
    uint8_t found_ports=0, avail_eths=rte_eth_dev_count_avail();

    for(uint32_t id=0; id<avail_eths && found_ports!=3; ++id) {
        rte_eth_dev_info_get(id, &dev_info);
        printf("Device %u info:\n", id);
        printf("  Driver name: %s\n", dev_info.driver_name);
        printf("  Min Rx bufsize: %u\n", dev_info.min_rx_bufsize);
        printf("  Max Rx pktlen: %u\n", dev_info.max_rx_pktlen);
        printf("  Max Rx queues: %u\n", dev_info.max_rx_queues);
        printf("  Max Tx queues: %u\n", dev_info.max_tx_queues);
        if(strcmp(dev_info.driver_name, "net_tap")==0&&!(found_ports&1)) {
            *tap_id=id;
            found_ports|=1;
        } else if((~found_ports)&2) {
            *trunk_id=id;
            found_ports|=2;
        }
        printf("found_ports: %d\n", found_ports);
    }

    return found_ports!=1;
}

int main(int ac, char *as[]) {
    running=1;
    if(ac==1) {
        fprintf(stderr, "Usage: %s [rules]\n", as[0]);
        return EXIT_FAILURE;
    }

    signal(SIGUSR1, print_stats);
    signal(SIGINT, exit_handler);
    signal(SIGKILL, exit_handler);
//    signal(SIGSEGV, exit_handler);

    int offset;

    if((offset=rte_eal_init(ac, as))<0)
        rte_exit(EXIT_FAILURE, "Error: could not init EAL.\n");
    ++offset;

    if(offset>=ac) {
        rte_exit(EXIT_FAILURE, "Usage: %s [[rte  arguments]...] [rules]\n", as[0]);
    }

    ac-=offset;
    as=as+offset;

    uint16_t avail_eths;
    struct rte_ether_addr tap_macaddr;
    if((avail_eths=rte_eth_dev_count_avail())<1)
        rte_exit(EXIT_FAILURE, "Error: not enough devices available.\n");

    if(find_tap_trunk_devs(&tap_port_id, &trunk_port_id))
        rte_exit(EXIT_FAILURE, "Error: could not find a tap/trunk port.\n");

    rte_eth_macaddr_get(tap_port_id, &tap_macaddr);

    if(setup_memory(&ext_mem, &mpool_payload)) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

#define RX_OC(X) RTE_ETH_RX_OFFLOAD_##X
#define TX_OC(X) RTE_ETH_TX_OFFLOAD_##X

    if(setup_port(trunk_port_id, &ext_mem, mpool_payload, DEFAULT_NB_QUEUES, DEFAULT_NB_QUEUES,
                  RX_OC(IPV4_CKSUM)|RX_OC(TCP_CKSUM)|RX_OC(UDP_CKSUM),
                  TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))
            |setup_port(tap_port_id, &ext_mem, mpool_payload, DEFAULT_NB_QUEUES, DEFAULT_NB_QUEUES,
                        RX_OC(IPV4_CKSUM)|RX_OC(TCP_CKSUM)|RX_OC(UDP_CKSUM),
                        TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

#undef RX_OC
#undef TX_OC

    port_stats=rte_malloc("port_stats", sizeof(stats_t)*2, 0);
    old_port_stats=rte_malloc("old_port_stats", sizeof(stats_t)*2, 0);
    memset(port_stats, 0, sizeof(stats_t)*2);
    memset(old_port_stats, 0, sizeof(stats_t)*2);

    memset(&ruleset, 0, sizeof(ruleset_t));

    if(!parse_ruleset(&ruleset, as[0])) {
        fprintf(stderr, "Error: could not parse ruleset \"%s\"\n", as[0]);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    printf("parsed ruleset \"%s\" with %lu rules\n", as[0], ruleset.num_rules);

    struct rte_bv_classifier_field_def fdefs[5];
    uint32_t fdefs_offsets[5]= {	offsetof(struct rte_ipv4_hdr, src_addr),
                                    offsetof(struct rte_ipv4_hdr, dst_addr),
                                    sizeof(struct rte_ipv4_hdr)+offsetof(struct rte_tcp_hdr, src_port),
                                    sizeof(struct rte_ipv4_hdr)+offsetof(struct rte_tcp_hdr, dst_port),
                                    offsetof(struct rte_ipv4_hdr, next_proto_id)
                               },
                               fdefs_sizes[5]= {4,4,2,2,1},
    ptype_masks[5] = {
        RTE_PTYPE_L2_MASK|RTE_PTYPE_L3_IPV4|RTE_PTYPE_L4_MASK,
        RTE_PTYPE_L2_MASK|RTE_PTYPE_L3_IPV4|RTE_PTYPE_L4_MASK,
        RTE_PTYPE_L2_MASK|RTE_PTYPE_L3_IPV4|RTE_PTYPE_L4_TCP|RTE_PTYPE_L4_UDP,
        RTE_PTYPE_L2_MASK|RTE_PTYPE_L3_IPV4|RTE_PTYPE_L4_TCP|RTE_PTYPE_L4_UDP,
        RTE_PTYPE_L2_ETHER|RTE_PTYPE_L3_MASK|RTE_PTYPE_L4_MASK
    };

    for(size_t i=0; i<5; ++i) {
        fdefs[i].offset=sizeof(struct rte_ether_hdr) + fdefs_offsets[i];
        fdefs[i].type=RTE_BV_CLASSIFIER_FIELD_TYPE_RANGE;
        fdefs[i].ptype_mask=ptype_masks[i];
        fdefs[i].size=fdefs_sizes[i];
    }

    struct rte_bv_classifier_params classifier_params = { .num_fields=5, .field_defs=fdefs, .num_rules=ruleset.num_rules };

    classifier=rte_bv_classifier_create(&classifier_params, rte_socket_id(), 1);

    if(classifier==NULL) {
        printf("classifier==NULL\n");
        goto err;
    }

    uint8_t **actions=rte_malloc("actions", ruleset.num_rules*sizeof(uint8_t *), sizeof(uint8_t *));
    for(uint32_t i=0; i<ruleset.num_rules; ++i)
        actions[i]=&ruleset.actions[i];

    if(rte_bv_classifier_entry_add_bulk(classifier, ruleset.rules, (void **) actions, ruleset.num_rules, NULL, NULL))
        goto err;

    free_ruleset_except_actions(&ruleset);

    struct timeval t;
    gettimeofday(&t, NULL);
    timestamp=1000000*t.tv_sec+t.tv_usec;

    trunk_tx_param_t trunk_tx_param= {.classifier=classifier, .tap_macaddr=&tap_macaddr};
    uint16_t coreid=rte_get_next_lcore(rte_get_main_lcore(), 1, 1);
    rte_eal_remote_launch(tap_tx, NULL, coreid);

    coreid=rte_get_next_lcore(coreid, 1, 1);
    rte_eal_remote_launch(trunk_rx, (void *) classifier, coreid);

    printf("FIREWALL READY\n");
    trunk_tx(&trunk_tx_param);

    rte_eal_mp_wait_lcore();

    rte_bv_classifier_free(classifier);

    free(ruleset.actions);

    rte_eal_cleanup();
    return EXIT_SUCCESS;

err:
    free_ruleset(&ruleset);
    rte_eal_cleanup();
    return EXIT_FAILURE;
}

#ifdef __cplusplus
}
#endif
