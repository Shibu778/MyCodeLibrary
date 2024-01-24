function Y=psr_deneme(x,m,tao,npoint)
%Phase space reconstruction
%x : time series 
%m : embedding dimension
%tao : time delay
%npoint : total number of reconstructed vectors
%Y : M x m matrix
% author:"Merve Kizilkaya"
N=length(x);
if nargin == 4
    M=npoint;
else
    M=N-(m-1)*tao;
end
Y=zeros(M,m); 
for i=1:m
    Y(:,i)=x((1:M)+(i-1)*tao)';
end