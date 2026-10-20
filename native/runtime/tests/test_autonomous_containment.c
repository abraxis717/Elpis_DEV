/* Launcher regression: a hostile child cannot make persistent files or modify
 * existing arbitrary files; only the four pre-provisioned ECS_C authority slots
 * (two records, two witnesses) are writable, and none past the record size.
 * Test-only files created in one temporary directory and removed on exit.
 */
#define _GNU_SOURCE 1
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define SLOT (1344 + 160 * 2) /* the authority record at its smallest capacity */
#define WITNESS 240

static int rw(const char *path, int flags) {
    int fd = open(path, flags, 0600);
    if (fd < 0) return -1;
    if (write(fd, "BAD", 3) != 3) { close(fd); return -1; }
    close(fd); return 0;
}
static int make_slot(const char *path, size_t size, char fill) {
    int fd = open(path, O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    char buf[SLOT]; memset(buf, fill, sizeof buf);
    int ok = write(fd, buf, size) == (ssize_t)size;
    close(fd); return ok ? 0 : -1;
}
static int check_slot(const char *path, size_t size, char expected) {
    char buf[SLOT + 1]; int fd = open(path, O_RDONLY|O_CLOEXEC);
    if (fd < 0) return -1;
    ssize_t n = read(fd, buf, sizeof buf);
    struct stat st; int ok = fstat(fd, &st) == 0 && st.st_size == (off_t)size && n == (ssize_t)size;
    close(fd);
    for (size_t i = 0; ok && i < size; ++i) if (buf[i] != expected) ok = 0;
    return ok ? 0 : -1;
}
static void pathcat(char *out,size_t cap,const char *dir,const char *file) {
    if (snprintf(out,cap,"%s/%s",dir,file) >= (int)cap) exit(2);
}
static int probe(const char *mode,const char *dir) {
    char a[4096], b[4096], c[4096], w[4096];
    pathcat(a,sizeof a,dir,"existing.txt");
    pathcat(b,sizeof b,dir,"new.txt");
    pathcat(c,sizeof c,dir,"state/authority.a");
    pathcat(w,sizeof w,dir,"authority/witness.a");
    if (!strcmp(mode,"read")) {
        int fd=open(a,O_RDONLY); char buf[2]={0};
        int ok=fd>=0 && read(fd,buf,1)==1 && buf[0]=='X';
        if(fd>=0) close(fd);
        return ok?0:1;
    }
    if (!strcmp(mode,"create")) return rw(b,O_CREAT|O_EXCL|O_WRONLY)==0?1:0;
    if (!strcmp(mode,"overwrite")) return rw(a,O_WRONLY)==0?1:0;
    if (!strcmp(mode,"append")) return rw(a,O_WRONLY|O_APPEND)==0?1:0;
    if (!strcmp(mode,"truncate")) return truncate(a,0)==0?1:0;
    if (!strcmp(mode,"rename")) return rename(a,b)==0?1:0;
    if (!strcmp(mode,"unlink")) return unlink(a)==0?1:0;
    if (!strcmp(mode,"mkdir")) return mkdir(b,0700)==0?1:0;
    if (!strcmp(mode,"hardlink")) return link(a,b)==0?1:0;
    if (!strcmp(mode,"symlink")) return symlink(a,b)==0?1:0;
    if (!strcmp(mode,"chmod")) return chmod(a,0666)==0?1:0;
    if (!strcmp(mode,"fd_escape")) return fcntl(8,F_GETFD)<0?0:1;
    if (!strcmp(mode,"slot_write")) {
        int fd=open(c,O_WRONLY); if(fd<0)return 1;
        int ok = pwrite(fd,"B",1,0)==1; close(fd); return ok?0:1;
    }
    if (!strcmp(mode,"slot_expand")) {
        signal(SIGXFSZ, SIG_IGN);
        int fd=open(c,O_WRONLY); if(fd<0)return 0;
        int ok = pwrite(fd,"B",1,SLOT)==1; close(fd); return ok?1:0;
    }
    if (!strcmp(mode,"witness_write")) {
        int fd=open(w,O_WRONLY); if(fd<0)return 1;
        int ok = pwrite(fd,"V",1,0)==1; close(fd); return ok?0:1;
    }
    if (!strcmp(mode,"witness_expand")) {
        signal(SIGXFSZ, SIG_IGN);
        int fd=open(w,O_WRONLY); if(fd<0)return 0;
        int ok = pwrite(fd,"V",1,SLOT)==1; close(fd); return ok?1:0;
    }
    if (!strcmp(mode,"witness_truncate")) return truncate(w,0)==0?1:0;
    if (!strcmp(mode,"slot_truncate")) return truncate(c,0)==0?1:0;
    if (!strcmp(mode,"slot_ftruncate")) {
        int fd=open(c,O_WRONLY); if(fd<0)return 0;
        int ok=ftruncate(fd,0)==0; close(fd); return ok?1:0;
    }
    return 2;
}
static int child(const char *runner, const char *self, const char *dir,
                 const char *mode, int permit) {
    pid_t pid = fork();
    if(pid<0)return -1;
    if(pid==0){
        if (!strcmp(mode,"fd_escape")) {
            char file[4096];pathcat(file,sizeof file,dir,"existing.txt");
            int saved=open(file,O_WRONLY);
            if (saved<0 || dup2(saved,8)!=8) _exit(127);
            if (saved!=8) close(saved);
        }
        if (!strcmp(mode,"stdio_escape")) {
            char file[4096];pathcat(file,sizeof file,dir,"existing.txt");
            int saved=open(file,O_WRONLY);
            if (saved<0 || dup2(saved,1)!=1) _exit(127);
            close(saved);
        }
        char state[4096],authority[4096];
        pathcat(state,sizeof state,dir,"state");
        pathcat(authority,sizeof authority,dir,"authority");
        if(permit)execl(runner,runner,"--authority",state,authority,"--",self,"probe",mode,dir,(char*)NULL);
        else execl(runner,runner,"--",self,"probe",mode,dir,(char*)NULL);
        _exit(127);
    }
    int status;
    if(waitpid(pid,&status,0)!=pid)return -1;
    return WIFEXITED(status)?WEXITSTATUS(status):-1;
}
int main(int argc,char **argv) {
    if(argc==4 && !strcmp(argv[1],"probe"))return probe(argv[2],argv[3]);
    if(argc!=2){fprintf(stderr,"usage: test_autonomous_containment LAUNCHER\n");return 2;}
    char dir[]="/tmp/elpis-autonomous-r0-XXXXXX";
    if(!mkdtemp(dir)){perror("mkdtemp");return 2;}
    char existing[4096],state[4096],authority[4096],a[4096],b[4096],wa[4096],wb[4096],newfile[4096];
    pathcat(existing,sizeof existing,dir,"existing.txt");
    pathcat(state,sizeof state,dir,"state");
    pathcat(authority,sizeof authority,dir,"authority");
    pathcat(a,sizeof a,dir,"state/authority.a");
    pathcat(b,sizeof b,dir,"state/authority.b");
    pathcat(wa,sizeof wa,dir,"authority/witness.a");
    pathcat(wb,sizeof wb,dir,"authority/witness.b");
    pathcat(newfile,sizeof newfile,dir,"new.txt");
    int fd=open(existing,O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0 || write(fd,"X",1)!=1){perror("create fixture");return 2;}
    close(fd);
    if(mkdir(state,0700)!=0 || mkdir(authority,0700)!=0){perror("directories");return 2;}
    if(make_slot(a,SLOT,'A')!=0 || make_slot(b,SLOT,'A')!=0 || make_slot(wa,WITNESS,'W')!=0 ||
       make_slot(wb,WITNESS,'W')!=0){perror("slots");return 2;}
    struct {const char *name;int permit;} tests[]={
        {"read",0},{"create",0},{"overwrite",0},{"append",0},
        {"truncate",0},{"rename",0},{"unlink",0},{"mkdir",0},
        {"hardlink",0},{"symlink",0},{"chmod",0},
        {"read",1},{"create",1},{"overwrite",1},
        {"fd_escape",1},{"stdio_escape",1},
        {"slot_write",1},{"slot_expand",1},
        {"slot_truncate",1},{"slot_ftruncate",1},{"chmod",1},
        {"witness_write",1},{"witness_expand",1},{"witness_truncate",1}
    };
    int errors=0;
    for(size_t j=0;j<sizeof tests/sizeof tests[0];j++){
        int rc=child(argv[1],argv[0],dir,tests[j].name,tests[j].permit);
        int expected = !strcmp(tests[j].name,"stdio_escape") ? 125 : 0;
        if(rc!=expected){fprintf(stderr,"FAIL %s %s rc=%d expected=%d\n",tests[j].name,
            tests[j].permit?"authority":"deny-all",rc,expected);errors++;}
        if(access(newfile,F_OK)==0 || access(existing,F_OK)!=0 ||
           check_slot(b,SLOT,'A')!=0 || check_slot(wb,WITNESS,'W')!=0){
            fprintf(stderr,"FAIL fixture after %s\n",tests[j].name);errors++;}
    }
    fd=open(existing,O_RDONLY);char byte=0;
    if(fd<0 || read(fd,&byte,1)!=1 || byte!='X')errors++;
    if(fd>=0)close(fd);
    fd=open(a,O_RDONLY);char first=0;
    if(fd<0 || read(fd,&first,1)!=1 || first!='B')errors++;
    if(fd>=0)close(fd);
    fd=open(wa,O_RDONLY);first=0;
    if(fd<0 || read(fd,&first,1)!=1 || first!='V')errors++;
    if(fd>=0)close(fd);
    struct stat st;
    if(stat(a,&st)!=0 || st.st_size!=SLOT || stat(wa,&st)!=0 || st.st_size!=WITNESS)errors++;
    if(unlink(a)||unlink(b)||unlink(wa)||unlink(wb)||rmdir(state)||rmdir(authority)||unlink(existing)||rmdir(dir)){
        perror("cleanup");errors++;}
    if(errors){fprintf(stderr,"ELPIS_AUTONOMOUS_CONTAINMENT_FAIL=%d\n",errors);return 1;}
    puts("ELPIS_AUTONOMOUS_CONTAINMENT_PASS");return 0;
}
