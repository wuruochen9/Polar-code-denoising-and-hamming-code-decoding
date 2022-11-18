import numpy as np
from data_generate import data_generation
import os
import torch
import torch.optim as optim
import torch.utils.data as Data
from torch import nn
from model import Encoder
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'
def make_data(data,label,src_vocab):#数据预处理
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(data)):
      da=data[i]
      la=label[i]
      enc_input = [src_vocab[n] for n in da]
      #print(co)
       # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
      dec_output=[tgt_vocab[n]for n in la]
      # dec_output.extend(tgt_vocab["P"] for n in range(8))

       # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
       # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
      # print(dec_input)
      
      enc_inputs.append(enc_input)
      dec_outputs.append(dec_output)
      
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_outputs)
    #

# 
class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_outputs = dec_outputs
  
  def __len__(self):
    return self.enc_inputs.shape[0]
  
  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_outputs[idx]
#data generation
do_train=1
do_data_gen=1
k = 8
N = 16
SNR = 3.0   #Eb/N0 单位dB from 3 to 10
if do_data_gen:
  data_generation(SNR,k,N)
data=np.load("encode_snr3.npy")
label=np.load("label16_snr3.npy")
max=0
min=10
for i in range(len(data)):
  for j in range(16):
    if data[i][j]>max:
      max=data[i][j]
    if data[i][j]<min:
      min=data[i][j]
print(min)
print(max)
# min=-33
# max=30
if -min>max:
    max=-min
    max=int(max)
    min=int(min)
    voc=np.zeros((-min)*2+1)
    for i in range((-min)*2+1):
        voc[i]=-(-min)+i

    # np.round(a,0)
    # voc=vocab(-min)
else:
    max=int(max)
    min=int(min)
    # voc=vocab(max) 
    voc=np.zeros((max)*2+1)
    for i in range((max)*2+1):
        voc[i]=-(max)+i
# print(voc)

#create vocabulary for src and target
# Padding Should be Zero
src_vocab = {}
for i in range(len(voc)):
  src_vocab[voc[i]]=i+1
src_vocab['-0.']=max*2+2

print(src_vocab)
src_vocab_size = len(src_vocab)

batch_size=32
tgt_vocab = {-1 : 0, 1 : 1}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 16 # enc_input max sequence length
tgt_len = 8 # dec_input(=dec_output) max sequence length
# Transformer Parameters
d_model = 1024  # Embedding Size
d_ff = 256 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V

n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
data=data.tolist()
label=label.tolist()
enc_inputs,  dec_outputs = make_data(data,label,src_vocab)

training_input=enc_inputs[int(len(enc_inputs)/2):]
training_label=dec_outputs[int(len(dec_outputs)/2):]
eval_input=enc_inputs[:int(len(enc_inputs)/4)]
eval_label=dec_outputs[:int(len(dec_outputs)/4)]
test_input=enc_inputs[int(len(enc_inputs)/4):int(len(enc_inputs)/2)]
test_label=dec_outputs[int(len(dec_outputs)/4):int(len(dec_outputs)/2)]
training_data=MyDataSet(training_input,training_label)
eval_data=MyDataSet(eval_input,eval_label)
test_data=MyDataSet(test_input,test_label)
train_loader = Data.DataLoader(training_data,batch_size,shuffle=True,num_workers=0, pin_memory=True)
dev_loader= Data.DataLoader(eval_data,batch_size,shuffle=True,num_workers=0, pin_memory=True)
test_loader=Data.DataLoader(test_data,batch_size,shuffle=False,num_workers=0, pin_memory=True)
model = Encoder(src_vocab_size,d_model,n_layers,tgt_vocab_size,d_k,n_heads,d_v,d_ff).cuda(2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

loss_list=[]
min_loss=10e9
MAX_EPOCH=300

#---------training-----------
if do_train:
  for epoch in range(MAX_EPOCH):
      train_acc_num=0
      train_loss=0
      for i,(enc_inputs, dec_outputs) in enumerate(train_loader):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        #print(dec_outputs.size())
        enc_inputs, dec_outputs = enc_inputs.cuda(2), dec_outputs.cuda(2)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs,_ = model(enc_inputs)
        # print(outputs.size())
        # print(dec_outputs[0])
        # outputs=outputs.cpu()
        # dec_outputs=dec_outputs.cpu()
        loss = criterion(outputs, dec_outputs.view(-1))
        train_loss += loss.item()
        pred = torch.max(outputs,1)[1]
        train_correct = (pred == dec_outputs.view(-1)).sum()
        train_acc_num += train_correct.item()
        if dec_outputs.size()[0]!=batch_size:
            train_acc = train_acc_num/(((i)*batch_size+dec_outputs.size()[0])*16)
        else:
            train_acc = train_acc_num/((i+1)*batch_size*16)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      train_loss = train_loss/((i+1))
      
      loss_list.append(train_loss)
      print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss),'acc =', '{:.6f}'.format(train_acc))
      eval_loss=0
      eval_acc_num=0
      for j, (enc_inputs, dec_outputs) in enumerate(dev_loader):
          enc_inputs, dec_outputs = enc_inputs.cuda(2), dec_outputs.cuda(2)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
          outputs,_ = model(enc_inputs)
          loss = criterion(outputs, dec_outputs.view(-1))
          eval_loss += loss.item()
          pred = torch.max(outputs,1)[1]
          eval_correct = (pred == dec_outputs.view(-1)).sum()
          eval_acc_num += eval_correct.item()
          if dec_outputs.size()[0]!=batch_size:
              eval_acc = eval_acc_num/(((j)*batch_size+dec_outputs.size()[0])*16)
          else:
              eval_acc = eval_acc_num/((j+1)*batch_size*16)
          lenth = j
              
      eval_loss = eval_loss/((lenth+1))
      if eval_loss<min_loss:
            min_loss=eval_loss
            torch.save(model,"/home/wrc/hcw/best.pth")
      print('eval Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss, eval_acc ))
      is_last=False
#-----------test--------------

load_from="/home/wrc/hcw/best.pth"
model=torch.load(load_from)
model=model.cuda(2)
test_acc_num=0
test_loss=0


for i, (enc_inputs, dec_outputs) in enumerate(test_loader):
    enc_inputs, dec_outputs = enc_inputs.cuda(2), dec_outputs.cuda(2)
    outputs,_ = model(enc_inputs)
    loss = criterion(outputs, dec_outputs.view(-1))
    test_loss += loss.item()
    pred = torch.max(outputs,1)[1]
    test_correct = (pred == dec_outputs.view(-1)).sum()
    test_acc_num += test_correct.item()
                        
    if dec_outputs.size()[0]!=batch_size:
        test_acc = test_acc_num/((i)*batch_size+dec_outputs.size()[0]*16)
    else:
        test_acc = test_acc_num/((i+1)*batch_size*16)
   
    lenth = i

test_loss = test_loss/((lenth+1))
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(test_loss, test_acc ))