<server.c>

#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>

#define port_num    45000
#define BUFSIZE     1024
#define m_len       1500
#define Thread_num  2

char message[m_len];
int cntnum = 0;
int resultnum = 0 ;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
char last[2][21];

void *sock_thread(void *arg){
    int new_sock = *((int *)arg);
    char message[BUFSIZE];
    long long int buf;
    read(new_sock,&last[resultnum],21);
    pthread_mutex_lock(&lock);
    printf("client send value = %s\n", last[resultnum]);
    resultnum ++;
    pthread_mutex_unlock(&lock);
    sleep(1);
    close(new_sock);
    return 0;
}


int main(int argc, char **argv){
    int server_sock, new_sock[Thread_num];
    struct sockaddr_in server_addr, client_addr;
    int i,status;
    socklen_t addrlen = sizeof(server_addr);

    pthread_t tid[Thread_num];
    pid_t pid;



    server_sock = socket(PF_INET,SOCK_STREAM,0);
    memset(&server_addr, 0 ,sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_num);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
   
    bind(server_sock,(struct sockaddr *) &server_addr, sizeof(server_addr));


    int sock,client_sock;
    struct sockaddr_in my_server_addr, my_client_addr;
    socklen_t new_addrlen = sizeof(my_client_addr);
    sock = socket(PF_INET,SOCK_STREAM,0);
    memset(&my_server_addr,0,sizeof(my_server_addr));
    my_server_addr.sin_family = AF_INET;
    my_server_addr.sin_port = htons(port_num+200);

    my_server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if((bind(sock,(struct sockaddr *)&my_server_addr, sizeof(my_server_addr)))==-1){
printf("bind error\n");
exit(1);
    }
    if(listen(sock,2) == -1){
printf("listen error\n");
exit(1);
    }
    client_sock = accept(sock,(struct sockaddr *)&my_client_addr, &new_addrlen);
    if(client_sock == -1){
perror("Accept error");
exit(1);
    }

    while(1){
        listen(server_sock, 10);
        printf("client wait...\n");
        new_sock[cntnum] = accept(server_sock,(struct sockaddr *)&client_addr, &addrlen);
        printf("accept client!\n");
        printf("Client IP address: %u, Port number: %d\r\n",client_addr.sin_addr.s_addr,ntohs(client_addr.sin_port));
        pthread_create(&tid[cntnum], NULL, &sock_thread, (void *) &new_sock[cntnum]);
        pthread_join(tid[cntnum],NULL);
        cntnum++;
        if(cntnum == Thread_num){
   write(client_sock, &last[0], 21);
   write(client_sock, &last[1], 21);
}
               
    }
    return 0;
   
}
