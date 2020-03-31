#include <stdio.h>
#include <pcap.h>

int pcnt = 0;

typedef struct ethernet_header{
    unsigned char dst_mac_addr[6];
    unsigned char src_mac_addr[6];
    unsigned short type;
}ethernet_header;

typedef struct ip_header{
    unsigned char hlen:4;
    unsigned char version:4;
    unsigned char service;
    unsigned short tlen;
    unsigned short id;
    unsigned short frag;
    unsigned char ttl;
    unsigned char protocol;
    unsigned short checksum;
    unsigned int src_ip_addr;
    unsigned int dst_ip_addr;
}ip_header;

typedef struct tcp_header{
    unsigned short src_port;
    unsigned short dst_port;
    unsigned int seq_num;
    unsigned int ack_num;
    unsigned char hlen;
    unsigned char flag;
    unsigned short win_size;
    unsigned short checksum;
    unsigned short urgent;
    unsigned char option[12];
}tcp_header;

void viewinfo(struct pcap_pkthdr *pcap_h,ethernet_header *ph,ip_header* ip,tcp_header* tcp){
    int i = 0;
    int j;
    int hh;
    int mm;
    int ss;
    unsigned int remaintime;
    unsigned short totallen;
    remaintime = pcap_h->ts.tv_sec;
    ss = remaintime % 60;
    remaintime /= 60;
    mm = remaintime % 60;
    remaintime /=60;
    hh = (remaintime+9) % 24;
    totallen = (ip->tlen & 0xff00) >> 8;
    totallen += (ip->tlen & 0xff) << 8;
    printf("Time:%02d:%02d:%02d.%06d\n",hh,mm,ss,pcap_h->ts.tv_usec);
    printf("captured length:%ubytes actual length:%ubytes \nIP header length:%ubytes, total length:%u",pcap_h->caplen,pcap_h->len,(ip->hlen)*4,totallen);
    printf("\n");

    printf("Destination mac address:");
    for(i = 0; i<6; i++){
        if(i == 5) {
            printf("%02x",ph->dst_mac_addr[i]);
            break;
        }
        printf("%02x:",ph->dst_mac_addr[i]);
    }
    printf("\n");
    printf("Source mac address:");
    for(i = 0; i<6; i++){
        if(i == 5){
            printf("%02x",ph->src_mac_addr[i]);
            break;
        }
        printf("%02x:",ph->src_mac_addr[i]);
    }
    printf("\n");

    unsigned short revers;
    printf("-----------------IP header------------------\n");

    revers = (ip->id & 0xff) << 8;
    revers += (ip->id & 0xff00) >> 8;
    printf("ID:%d\n",revers);
    printf("Fragment:");
    if(ip->frag & 0x40){
        printf("DF");
    }
    else if(ip->frag & 0x20){
        printf("MF");
    }
    printf("\n");
    printf("Source IP address:%d.",ip->src_ip_addr & 0xff);
    printf("%d.",ip->src_ip_addr>>8 & 0xff);
    printf("%d.",ip->src_ip_addr>>16 & 0xff);
    printf("%d\n",ip->src_ip_addr>>24 & 0xff);
    printf("Destination IP address:%d.",ip->dst_ip_addr & 0xff);
    printf("%d.",ip->dst_ip_addr>>8 & 0xff);
    printf("%d.",ip->dst_ip_addr>>16 & 0xff);
    printf("%d\n",ip->dst_ip_addr>>24 & 0xff);
    printf("Protocol:");
    switch(ip->protocol){
        case 1: printf("ICMP\n"); break;
        case 6: printf("TCP\n"); break;
        case 17: printf("UDP\n"); break;
        default: break; 
    }

    printf("TTL:%d\n",ip->ttl);
    printf("-----------------TCP header------------------\n");
    int payload = 0;
    payload = (ip->tlen & 0xff) << 8;
    payload += (ip->tlen & 0xff00) >> 8;
    payload = payload - ip->hlen*4 - ((tcp->hlen & 0xf0) >> 4)*4 ;
    revers = (tcp->src_port & 0xff) << 8;
    revers += (tcp->src_port & 0xff00) >> 8;
    printf("Source Port:%d\n",revers);
    revers = (tcp->dst_port & 0xff) << 8;
    revers += (tcp->dst_port & 0xff00) >> 8;
    printf("Destination Port:%d\n",revers);
    //starting seqnum ending seqnum
    //Acknowledgement number
    unsigned int change;
    change = (tcp->seq_num & 0xff) << 24;
    change += (tcp->seq_num & 0xff00) << 8;
    change += (tcp->seq_num & 0xff0000) >> 8;
    change += (tcp->seq_num & 0xff000000) >> 24;
    printf("Starting Sequence number:%u\n",change);
    printf("Ending Sequence number:%u\n",change+payload);   
    change = (tcp->ack_num & 0xff) << 24;
    change += (tcp->ack_num & 0xff00) << 8;
    change += (tcp->ack_num & 0xff0000) >> 8;
    change += (tcp->ack_num & 0xff000000) >> 24;
    printf("Acknowledgment number:%u\n",change);
    //Tcp payload size
    printf("Payload size:%dbytes\n",payload);
    //Tcp header size
    revers = (tcp->hlen & 0xf0) >> 4;
    if(revers == 0) return;
    printf("Tcp header size:%dbytes(%d)\n",revers*4,revers);
    int thlen = revers;
    //Advertising window size
    revers = (tcp->win_size & 0xff) <<8;
    revers += (tcp->win_size & 0xff00) >> 8;
    printf("Window size:%d\n",revers);
    //Tcp segment type
    printf("flag:");
    if(tcp->flag & 0x01) printf("F ");
    if(tcp->flag & 0x02) printf("S ");
    if(tcp->flag & 0x04) printf("R ");
    if(tcp->flag & 0x08) printf("P ");
    if(tcp->flag & 0x10) printf("A ");
    if(tcp->flag & 0x20) printf("U ");
    printf("\n");
    //Checksum value
    revers = (tcp->checksum & 0xff) << 8;
    revers += (tcp->checksum & 0xff00) >> 8;
    printf("check sum:0x%x\n",revers);
    int mss = 0;
    int lefte = 0;
    int righte = 0;
    //Tcp options if exists
    if(thlen > 5){
        i = 0;
        printf("Option:\n");
        while(1){
            if(i > 12) break;
            if(tcp->option[i] == 1){
                i++;
                printf("No-Operation\n");
                continue;
            }
            else if(tcp->option[i] == 2){
                i++;
                i++;
                mss += tcp->option[i];
                i++;
                mss = mss << 8;
                mss += tcp->option[i];
                printf("Maximum segment size:%dbytes\n",mss);
                i++;
                continue;
            }
            else if(tcp->option[i] == 3){
                i++;
                i++;
                printf("Window scale:%d\n",tcp->option[i]);
                i++;
                continue;
            }
            else if(tcp->option[i] == 4){
                printf("SACK permitted\n");
                i++;
                i++;
                continue;

            }
            else if(tcp->option[i] == 5){
                i++;
                i++;
                lefte = 0;
                righte = 0;
                for(j = 0; j<4 ; j++){
                    lefte <<= 8;
                    lefte += tcp->option[i];
                    i++;
                }
                for(j = 0; j<4 ; j++){
                    righte <<= 8;
                    righte += tcp->option[i];
                    i++;
                }
                printf("SACK %d-%d\n",lefte,righte);

            }
            i++;
        }
    }
}

int main(){
    char *err;
    pcap_t *f;
    struct pcap_pkthdr *header;
    const u_char *pack;
    f = pcap_open_offline("hwpacket.pcap",err);
    while(pcap_next_ex(f,&header,&pack) == 1){
        pcnt++;
        printf("%d------------------------\n",pcnt);
        ethernet_header *ph = (ethernet_header*)pack;
        ip_header *ip = (ip_header*)(pack+sizeof(ethernet_header));
        tcp_header *tcp = (tcp_header*)(pack+sizeof(ethernet_header)+sizeof(ip_header));
        viewinfo(header,ph,ip,tcp);
    }
    pcap_close(f);
    return 0;
}
