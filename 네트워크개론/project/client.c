#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>

#define BUFSIZE 1024
#define port_num 50000

int main(){
    int sock;
    int n;
    struct sockaddr_in6 server_addr;
    int buf;
    socklen_t addrlen = sizeof(server_addr);
    int server_port;
    int i,j,l;
    char message[BUFSIZE];
    sock = socket(AF_INET6,SOCK_STREAM,0);
    memset(&server_addr,0,sizeof(server_addr));
    server_addr.sin6_family = AF_INET6;
    server_addr.sin6_flowinfo = 0;
    server_addr.sin6_port = htons(port_num);
    inet_pton(AF_INET6, "2001:0000:53aa:064c:1861:78db:8c6e:5548", &server_addr.sin6_addr);
   
    if(connect(sock,(struct sockaddr*)&server_addr,sizeof(server_addr)) == -1){
        printf("connection error\n");
        exit(1);
    }
   
    int new_sock;
    struct sockaddr_in my_server_addr;
    long long int sum;

    socklen_t new_addrlen = sizeof(my_server_addr);
    int my_server_port;
    new_sock = socket(AF_INET,SOCK_STREAM,0);
    memset(&my_server_addr,0,sizeof(my_server_addr));
    my_server_addr.sin_family = AF_INET;
    my_server_addr.sin_port = htons(45200);
    inet_pton(AF_INET, "127.0.0.1", &my_server_addr.sin_addr);
    if(connect(new_sock,(struct sockaddr*)&my_server_addr,sizeof(my_server_addr)) == -1){
printf("connection error\n");
exit(1);
    }
    read(sock,message,BUFSIZE);
    printf("%s\n",message);
    printf("");
    char last[2][21];
    for(i = 0; i<6; i++){
read(sock,message,BUFSIZE);
printf("%s",message);
memset(message,0,sizeof(message));
read(0, message, BUFSIZE);
message[strlen(message)] = 0x0a;
write(sock,message,strlen(message));
memset(message,0,sizeof(message));
if(i == 5){
   
   read(new_sock,&last[0],21);
   write(sock,&last[0],21);
   read(new_sock,&last[1],21);
   write(sock,&last[1],21);
   close(new_sock);
   memset(message,0,BUFSIZE);
   read(sock,message,BUFSIZE);
   printf("%s",message);
   read(sock,message,BUFSIZE);
   printf("%s",message);
   read(sock,message,BUFSIZE);
   printf("%s",message);
}
else{
   read(sock,message,BUFSIZE);
   printf("%s",message);
   memset(message,0,sizeof(message));
}
    }

    close(sock);
   
    return 0;
}
