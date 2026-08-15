"""TileLang L3 #11 VGG16."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn.functional as F
from _l3_kernels import cvr,pool,flr,lr,ln
P1=lambda x:F.pad(x,(1,1,1,1));S=torch.npu.synchronize

def run(x,p):
    BS=x.shape[0];P=p
    c1w,c1b,c2w,c2b,c3w,c3b,c4w,c4b,c5w,c5b,c6w,c6b,c7w,c7b,c8w,c8b=P[:16]
    c9w,c9b,c10w,c10b,c11w,c11b,c12w,c12b,c13w,c13b=P[16:26]
    fc1w,fc1b,fc2w,fc2b,fc3w,fc3b=P[26:]
    h=P1(x);h=cvr(BS,3,64,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c1w,c1b);S()
    h=P1(h);h=cvr(BS,64,64,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c2w,c2b);S()
    h=pool(BS,64,h.shape[2],h.shape[3],2,2,TH=4,TW=4)(h);S()
    h=P1(h);h=cvr(BS,64,128,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c3w,c3b);S()
    h=P1(h);h=cvr(BS,128,128,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c4w,c4b);S()
    h=pool(BS,128,h.shape[2],h.shape[3],2,2,TH=4,TW=4)(h);S()
    h=P1(h);h=cvr(BS,128,256,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c5w,c5b);S()
    h=P1(h);h=cvr(BS,256,256,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c6w,c6b);S()
    h=P1(h);h=cvr(BS,256,256,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c7w,c7b);S()
    h=pool(BS,256,h.shape[2],h.shape[3],2,2,TH=4,TW=4)(h);S()
    h=P1(h);h=cvr(BS,256,512,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c8w,c8b);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c9w,c9b);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c10w,c10b);S()
    h=pool(BS,512,h.shape[2],h.shape[3],2,2,TH=4,TW=4)(h);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c11w,c11b);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c12w,c12b);S()
    h=P1(h);h=cvr(BS,512,512,h.shape[2],h.shape[3],3,1,TH=4,TW=4)(h,c13w,c13b);S()
    h=pool(BS,512,h.shape[2],h.shape[3],2,2,TH=4,TW=4)(h);S()
    h=flr(BS,512,h.shape[2],h.shape[3],fc1w.shape[0])(h,fc1w,fc1b);S()
    h=lr(BS,fc1w.shape[0],fc2w.shape[0])(h,fc2w,fc2b);S()
    return ln(BS,fc2w.shape[0],fc3w.shape[0])(h,fc3w,fc3b)

if __name__=="__main__":
    torch.manual_seed(0);BS,NC=2,10;IW=32
    x=torch.randn(BS,3,IW,IW).npu()
    # compute final spatial: 5x pool(2,2) from 32 -> 16 -> 8 -> 4 -> 2 -> 1
    FS=1
    # 13 conv weights: all 3x3, channels: 3->64->64, 64->128->128, 128->256->256->256, 256->512->512->512, 512->512->512->512
    ch=[(3,64),(64,64),(64,128),(128,128),(128,256),(256,256),(256,256),
        (256,512),(512,512),(512,512),(512,512),(512,512),(512,512)]
    P=[]
    for ic,oc in ch:
        P.append(torch.randn(oc,ic,3,3).npu())
        P.append(torch.randn(oc).npu())
    P.append(torch.randn(4096,512*FS*FS).npu());P.append(torch.randn(4096).npu())
    P.append(torch.randn(4096,4096).npu());P.append(torch.randn(4096).npu())
    P.append(torch.randn(NC,4096).npu());P.append(torch.randn(NC).npu())
    out=run(x,P)
    cp=[p.cpu() for p in P];xc=x.cpu()
    ops=[xc]
    for i in range(0,26,2):
        ops.append(F.relu(F.conv2d(F.pad(ops[-1],(1,1,1,1)) if i%4==0 else ops[-1],cp[i],cp[i+1],padding=1 if i>0 else 0)))
        if i in (2,6,10,18,24):ops.append(F.max_pool2d(ops[-1],2))
    # actually the loop logic is wrong - let me just manually trace
    # conv1: pad -> conv -> relu
    r=F.pad(xc,(1,1,1,1));r=F.relu(F.conv2d(r,cp[0],cp[1]))
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[2],cp[3]))
    r=F.max_pool2d(r,2)
    # Block2
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[4],cp[5]))
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[6],cp[7]))
    r=F.max_pool2d(r,2)
    # Block3
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[8],cp[9]))
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[10],cp[11]))
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[12],cp[13]))
    r=F.max_pool2d(r,2)
    # Block4
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[14],cp[15]))
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[16],cp[17]))
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[18],cp[19]))
    r=F.max_pool2d(r,2)
    # Block5: last conv before pool
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[20],cp[21]))
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[22],cp[23]))
    r=F.pad(r,(1,1,1,1));r=F.relu(F.conv2d(r,cp[24],cp[25]))
    r=F.max_pool2d(r,2)
    r=r.reshape(BS,-1);r=F.relu(F.linear(r,cp[26],cp[27]))
    r=F.relu(F.linear(r,cp[28],cp[29]));r=F.linear(r,cp[30],cp[31])
    torch.testing.assert_close(out.cpu(),r,rtol=1e-2,atol=1e-2)
    print("level3_011_vgg16 passed")
