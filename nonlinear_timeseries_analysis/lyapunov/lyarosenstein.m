function d = lyarosenstein(x,m,tao,meanperiod,maxiter) 
% d:divergence of nearest trajectoires
% x:signal
% tao:time delay
% m:embedding dimension
N=length(x);
M=N-(m-1)*tao;
Y=psr_deneme(x,m,tao);
for i=1:M
    i
    x0=ones(M,1)*Y(i,:);
    distance=sqrt(sum((Y-x0).^2,2));
    for j=1:M
        if abs(j-i)<=meanperiod
            distance(j)=1e10;
        end
    end
    [neardis(i) nearpos(i)]=min(distance);
end
for k=1:maxiter
    k
    maxind=M-k;
    evolve=0;
    pnt=0;
    for j=1:M
        if j<=maxind && nearpos(j)<=maxind
            dist_k=sqrt(sum((Y(j+k,:)-Y(nearpos(j)+k,:)).^2,2));
             if dist_k~=0
                evolve=evolve+log(dist_k);
                pnt=pnt+1;
             end
        end
    end
    if pnt > 0
        d(k)=evolve/pnt;
    else
        d(k)=0;
    end
    
end