"""TileLang L3 #12 VGG19."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cvr,pool,flr,lr,ln
P1=lambda x:F.pad(x,(1,1,1,1));S=torch.npu.synchronize

def run(x,p):
    BS=x.shape[0];P=p
    tks={'TH':4,'TW':4}
    h=P1(x);h=cvr(BS,3,64,h.shape[2],h.shape[3],3,1,**tks)(h,P[0],P[1]);S()
    h=P1(h);h=cvr(BS,64,64,h.shape[2],h.shape[3],3,1,**tks)(h,P[2],P[3]);S()
    h=pool(BS,64,h.shape[2],h.shape[3],2,2,**tks)(h);S()
    h=P1(h);h=cvr(BS,64,128,h.shape[2],h.shape[3],3,1,**tks)(h,P[4],P[5]);S()
    h=P1(h);h=cvr(BS,128,128,h.shape[2],h.shape[3],3,1,**tks)(h,P[6],P[7]);S()
    h=pool(BS,128,h.shape[2],h.shape[3],2,2,**tks)(h);S()
    h=P1(h);h=cvr(BS,128,256,h.shape[2],h.shape[3],3,1,**tks)(h,P[8],P[9]);S()
    h=P1(h);h=cvr(BS,256,256,h.shape[2],h.shape[3],3,1,**tks)(h,P[10],P[11]);S()
    h=P1(h);h=cvr(BS,256,256,h.shape[2],h.shape[3],3,1,**tks)(h,P[12],P[13]);S()
    h=P1(h);h=cvr(BS,256,256,h.shape[2],h.shape[3],3,1,**tks)(h,P[14],P[15]);S()
    h=pool(BS,256,h.shape[2],h.shape[3],2,2,**tks)(h);S()
    h=P1(h);h=cvr(BS,256,512,h.shape[2],h.shape[3],3,1,**tks)(h,P[16],P[17]);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,**tks)(h,P[18],P[19]);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,**tks)(h,P[20],P[21]);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,**tks)(h,P[22],P[23]);S()
    h=pool(BS,512,h.shape[2],h.shape[3],2,2,**tks)(h);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,**tks)(h,P[24],P[25]);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,**tks)(h,P[26],P[27]);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,**tks)(h,P[28],P[29]);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,**tks)(h,P[30],P[31]);S()
    h=pool(BS,512,h.shape[2],h.shape[3],2,2,**tks)(h);S()
    h=flr(BS,512,h.shape[2],h.shape[3],P[32].shape[0])(h,P[32],P[33]);S()
    h=lr(BS,P[32].shape[0],P[34].shape[0])(h,P[34],P[35]);S()
    return ln(BS,P[34].shape[0],P[36].shape[0])(h,P[36],P[37])

if __name__=="__main__":
    torch.manual_seed(0);BS,NC,FS=2,10,1
    x=torch.randn(BS,3,32,32).npu()
    ch=[(3,64),(64,64),(64,128),(128,128),(128,256),(256,256),(256,256),(256,256),
        (256,512),(512,512),(512,512),(512,512),(512,512),(512,512),(512,512),(512,512)]
    P=[];P2=[]
    for ic,oc in ch:P.append(torch.randn(oc,ic,3,3).npu());P.append(torch.randn(oc).npu())
    P.append(torch.randn(4096,512*FS*FS).npu());P.append(torch.randn(4096).npu())
    P.append(torch.randn(4096,4096).npu());P.append(torch.randn(4096).npu())
    P.append(torch.randn(NC,4096).npu());P.append(torch.randn(NC).npu())
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    def b(r,cw,cb):return F.relu(F.conv2d(F.pad(r,(1,1,1,1)),cw,cb))
    r=b(xc,cp[0],cp[1]);r=b(r,cp[2],cp[3]);r=F.max_pool2d(r,2)
    r=b(r,cp[4],cp[5]);r=b(r,cp[6],cp[7]);r=F.max_pool2d(r,2)
    r=b(r,cp[8],cp[9]);r=b(r,cp[10],cp[11]);r=b(r,cp[12],cp[13]);r=b(r,cp[14],cp[15]);r=F.max_pool2d(r,2)
    r=b(r,cp[16],cp[17]);r=b(r,cp[18],cp[19]);r=b(r,cp[20],cp[21]);r=b(r,cp[22],cp[23]);r=F.max_pool2d(r,2)
    r=b(r,cp[24],cp[25]);r=b(r,cp[26],cp[27]);r=b(r,cp[28],cp[29]);r=b(r,cp[30],cp[31]);r=F.max_pool2d(r,2)
    r=r.reshape(BS,-1);r=F.relu(F.linear(r,cp[32],cp[33]));r=F.relu(F.linear(r,cp[34],cp[35]));r=F.linear(r,cp[36],cp[37])
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_012_vgg19 passed")
