import numpy as np
from polarcodes import *
import tensorflow.keras.backend as K
def generate_input_data(k):#载入训练数据
    X = []
    for i in range(2**k):
        bin_str = bin(i)[2:].zfill(k)
        x = []
        for j in range(k):
            x.append(int(bin_str[j]))
        X.append(x)
    return np.array(X)


def get_index_set(N, k, snr_db):
    snr = 10**(snr_db/10)
    bhattacharya_param = np.exp(-snr)

    leaves = np.zeros(N)
    leaves[0] = bhattacharya_param
    for level in range(1, int(np.log2(N))+1):
        for i in range(0, int( (2**level)/2 )):
            p = leaves[i]
            leaves[i] = 2*p - p**2
            leaves[i + int( (2**level)/2 )] = p**2
    J = np.argsort(leaves)[:k]
    I = []
    for j in J:
        bin_str = bin(j)[2:].zfill(int(np.log2(N)))
        bin_str = bin_str[::-1]
        bin_str = '0b' + bin_str
        I.append(int(bin_str,2))
    return sorted(I)

def _polar_encode(u):

    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0,stages):
        i = 0
        while i < N:
            for j in range(0,n):
                idx = i+j
                x[idx] = x[idx] ^ x[idx+n]
            i=i+2*n
        n=2*n
    return x

def polar_encode(input, N, k, snr_db):
    indices = get_index_set(N, k ,snr_db)
    X = np.zeros((len(input), N),dtype=bool)
    X[:, indices] = input
    encoded = []

    for u in X:
        encoded.append(_polar_encode(u))
    return encoded

def bpsk_modulation(x):
    return -2*x + 1
    
def awgn_noise(x, noise_var):
    return x + K.random_normal(K.shape(x), mean = 0.0, stddev = np.sqrt(noise_var))
    
def llr(x, noise_var):
    return 2*x/noise_var
def data_generation(SNR,k,N):
#u_messages：未编码：k位
    u_messages = np.array(generate_input_data(k), dtype = np.float32)
    #print(len(u_messages))
    #x:编好码 N位
    x = np.array(polar_encode(u_messages, N, k, -3), np.float32)
    bpsk=x
    bpsk=bpsk_modulation(bpsk)
    #增加数据数量至1024个码
    # noise_var = 1.0/(2* 10**( (0 + 10*np.log10(0.01) )/10.0) )
    noise_var = 10**(-SNR/(20*(k/N)))
    for i in range(5):
        x=np.concatenate((x,x))
    x=x[:8192]
    #对数据进行信道损耗模拟
    x = bpsk_modulation(x)
    #print(x[0])
    x = awgn_noise(x, noise_var)
    x=llr(x,noise_var)
    #print(x[0])
    x=np.around(x,0)
    #print(len(u_messages))
    #print(x[1])
    #print(len(x))
    #算欧氏距离,改变误差过大数据
    u_m=[]
    x_b=[]
    for i in range(len(x)):
        min=100000
        for j in range(len(bpsk)):
            dis=0
            for l in range(N):
                dis=dis+(bpsk[j][l]-x[i][l])**2
            if dis<min:
                min=dis
                x_match_b=bpsk[j]
                
        # u_m.append(u_messages[ind])
        x_b.append(x_match_b)
    # u_m_np=np.array(x_b)
    x_label=np.array(x_b)#16 bits
    x_np=np.array(x)
    np.save("encode_snr3.npy",x_np)
    np.save("label16_snr3.npy",x_label)
    return 1
    


    
def vocab(h):
    a=np.zeros(h*2+1)
    for i in range(h*2+1):
        a[i]=-h+i

    np.round(a,0)
    s=[]
#print(a)
    for i in range(h*2+1):
        st=str(a[i])
        if st[0]=='-':
            if len(st)>5:
                if st[2]=='.':
                    if st[3]=='0':
                        st=st[0:2]
                    else:
                        t=int(st[1])
                        t=t+1
                        st=st[0]
                        st=st+str(t)
                        st=st+'.'
                
                if st[3]=='.':
                    if st[4]=='0':
                        st=st[0:3]
                    else:
                        if st[4]=='9':
                            t=int(st[2])
                            if t==9:
                                t1=int(st[1])
                                t1=t1+1
                                st=st[0]
                                st=st+str(t1)+'0'+'.'
                        
                            else:
                                t=t+1
                                st=st[0]
                                st=st+str(t)
                                st=st+'.'
                
        else:
            if len(st)>5:
                if st[1]=='.':
                    if st[2]=='0':
                        st=st[0:1]
                    else:
                        t=int(st[0])
                        t=t+1
                        st=''
                        st=st+str(t)
                        st=st+'.'

                elif st[2]=='.':
                    if st[3]=='0':
                        st=st[0:3]
                    else:
                        t=int(st[1])
                        if t==9:
                            t1=int(st[0])
                            t1=t1+1
                            st=str(t1)+'0'+'.'
                        else:
                            t=t+1
                            st=''
                            st=st+str(t)
                            st=st+'.'
                

            
        if(st[-1]=='0'):
            st=st[:-1]
            if(st[-1]=='0'):
                st=st[:-1]
        # st=st[:-1]
        s.append(st)
    return s

    