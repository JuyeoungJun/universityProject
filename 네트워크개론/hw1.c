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

void viewinfo(struct pcap_pkthdr *pcap_h,ethernet_header *ph,ip_header* ip){
	int i = 0;
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
		viewinfo(header,ph,ip);
	}
	pcap_close(f);
	return 0;
}
	
