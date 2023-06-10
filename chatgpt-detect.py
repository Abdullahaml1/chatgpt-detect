import argparse
import torch
import torch.nn as nn
import numpy as np
import random

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import json
import time
import nltk
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names

SEED = 1

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

kUNK = '<unk>'
kPAD = '<pad>'
kPAD_IND = 0 # index of padding token
kUNK_IND = 1

def class_labels(classes_l):
    '''
   :param classes_l: list of class names, numbers, ...
   :return class2ind(dict), ind2class(list)
   '''
    class2ind = {}
    ind2class = []
    for i in range(len(classes_l)):
        assert classes_l[i] not in class2ind, f'{classes_l[i]} is doblicated' 
        class2ind[classes_l[i]]= i
        ind2class.append(classes_l[i])

    return class2ind, ind2class


# You don't need to change this funtion
def load_data():
    """
    """
    dataset = load_dataset('Hello-SimpleAI/HC3', name='all', split='train')
    dataset = dataset.with_format('torch')
    return dataset



def load_words(exs):
    """
    vocabuary building
    Keyword arguments:
    exs: list of input questions-type pairs
    """
    pass



class UnpackedDataset(torch.utils.data.Dataset):
      """
      Pytorch data class for questions
      """
  
      def __init__(self, dataset, class2ind):
            self.samples = []

            for item in dataset:
                for c in class2ind:
                    if len(item[c]) >0:
                        s = {
                            'id': item['id'],
                            'question': item['question'],
                            'source': item['source'],
                            'answer': item[c][0],
                            'label': class2ind[c]
                        }
                        self.samples.append(s)
          
          
      ###You don't need to change this funtion
      def __getitem__(self, index):
          return self.samples[index]
            
      
      ###You don't need to change this funtion
      def __len__(self):
          return len(self.samples)


    

# batch function
class Batchify(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        :param batch (batch)
        """
        ids = []
        questions = []
        sources = []
        labels = []
        ans = []
        for item in batch:
            ids.append( item['id'])
            questions.append(item['question'])
            sources.append(item['source'])
            ans.append(item['answer'])
            labels.append(item['label'])

        ans_tokens = self.tokenizer(ans, padding='max_length',
                                    truncation=True,
                                    return_tensors="pt")
        labels_ten = torch.tensor(labels, dtype=torch.long)

        return{'ans_tokens': ans_tokens,
               'questions': questions,
               'sources': sources,
               'labels': labels_ten}



# evalute funciton
def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """
                  
    model.to(device)
    model.eval()
    num_examples = 0
    error = 0                              
   
    total_loss = 0.0
    num_examples = 0
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            ans_tokens = batch['ans_tokens']
            ans_tokens.to(device)
            labels = batch['labels']
      
              
            logits = model(**ans_tokens) # shape [batch x num_classes]
            print(type(logits))
            print(logits.shape)
            top_n, top_i = logits.topk(1)
            num_examples += ans_tokens.size(0)
            error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
   
   
        # Accuracy
        accuracy = 1 - error / num_examples
        avg_loss = total_loss/num_examples
    # print(f'Dev accuracy={accuracy:f}, Dev average Loss={avg_loss:f}')
    return accuracy


def train(args, model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model
    Keyword arguments:
    args: arguments 
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """

    pass




class ChatgptDetectModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """



    def __init__(self, n_classes, vocab_size, emb_dim=50,
                 n_hidden_units=50, nn_dropout=.5, padding_idx=0,
                 glove_embeddings=None):
        super().__init__()
        pass
        
        
       
    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass, returns the logits of the predictions.
        
        Keyword arguments:
        input_text : vectorized question text  [batch x questin_list]
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """

        pass

def show_error_samples(data_loader, model, loss_fun,
        ind2word_arr, ind2class, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    pass

''' ploting function '''
def plot_model(train, test, num_epochs):
    '''
    plots a given train and test data
    :param ax: matplotlib ax
    :param title: str
    :param train: train list
    :param test: test list
    :param test_point: number
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    def plot_val(ax, title, train, test, num_epochs=num_epochs):
        # x = np.arange(1, len(test)+1, 1) * (num_epochs / len(test))
        x = np.arange(len(test)) * ((num_epochs-1) / (len(test)-1))
        ax.plot(x, train, label='train', color='r')
        ax.plot(x, test, label='dev', color='b')
        # ax.plot([len(test)], [test_point], 'g*')
        # ax.annotate(f"test {title}={test_point:.3f}", xy=(len(test), test_point), xytext=(len(test)-1, test_point-.05))
        ax.set_xlabel('Epochs')
        ax.legend()
        ax.set_title(title)
        ax.grid()

    plot_val(ax1, 'Accuracy', train['accuracy'], test['accuracy']) 
    plot_val(ax2, 'Loss', train['loss'], test['loss']) 
    return fig, (ax1, ax2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--train-file', type=str, default='data/question_train_cl1.json')
    parser.add_argument('--dev-file', type=str, default='data/question_dev_cl1.json')
    parser.add_argument('--test-file', type=str, default='data/question_test_cl1.json')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--grad-clipping', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--save-model', type=str, default='q_type.pt')
    parser.add_argument('--load-model', type=str, default='q_type.pt')
    parser.add_argument("--limit", help="Number of training documents", type=int, default=-1, required=False)
    parser.add_argument('--checkpoint', type=int, default=21)
    parser.add_argument("--num-workers", help="Number of workers", type=int, default=2, required=False)
    parser.add_argument('--show-dev-error-samples', action='store_true', help='Print Error Dev samples', default=False)

    args = parser.parse_args()
    #### check if using gpu is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # model name
    tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
    model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta", torchscript=True)

    # batchify function
    batchify = Batchify(tokenizer)

    ### Load data
    dataset = load_data()
    print('Lenght of dataset', len(dataset))



    #get class to int mapping
    class2ind, ind2class = class_labels(['human_answers', 'chatgpt_answers'])  
    print('Number of Classes=', len(class2ind))

    if args.test:
        print('start Testing ..........')
        #### Load batchifed dataset
        test_dataset = UnpackedDataset(dataset, class2ind)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size,
                                                sampler=test_sampler,
                                                num_workers=args.num_workers,
                                                collate_fn=batchify)
        acc = evaluate(test_loader, model, device)
        print(f'Test Accuracy={acc:f}')

    # # show Error Dev Samples
    # elif args.show_dev_error_samples:
    #     model = torch.load(args.load_model)
    #     model.to(device)

    #     dev_dataset = QuestionDataset(dev_exs, word2ind, num_classes, class2ind)
    #     dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    #     dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
    #                                            sampler=dev_sampler,
    #                                            num_workers=args.num_workers,
    #                                            collate_fn=batchify)
    #     # Applying Devset
    #     dev_acc, dev_loss = show_error_samples(dev_loader, model,
    #                     nn.CrossEntropyLoss(),
    #                     np.array(ind2word), ind2class,
    #                     device)
    #     print(f'Dev acc={dev_acc:f}, dev_error={dev_loss:f}')

    # else:
    #     if args.resume:
    #         print('Resuming.....')
    #         model = torch.load(args.load_model)
    #     else:
    #         if args.use_glove:
    #             model = DanModel(num_classes, len(voc),
    #                     glove_embeddings=glove_embeddings,
    #                     padding_idx=word2ind[kPAD])
    #         else:
    #             model = DanModel(num_classes, len(voc))

    #         model.to(device)
    #     print(model)
    #     #### Load batchifed dataset
    #     train_dataset = QuestionDataset(train_exs, word2ind, num_classes, class2ind)
    #     train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    #     ''' Debug Print '''
    #     # train_sample = next(iter(train_dataset))
    #     # print('Debug DebugDebugDebugDebug----------------')
    #     # print(train_sample[0])
    #     # print(train_sample[1])

    #     dev_dataset = QuestionDataset(dev_exs, word2ind, num_classes, class2ind)
    #     dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    #     dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
    #                                            sampler=dev_sampler,
    #                                            num_workers=args.num_workers,
    #                                            collate_fn=batchify)

    #     ''' Debug Print '''
    #     # train_loader_debug = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #     #                                       sampler=train_sampler,
    #     #                                       num_workers=args.num_workers,
    #     #                                       collate_fn=batchify)
    #     # train_sample = next(iter(train_loader_debug))
    #     # print('Debug DebugDebugDebugDebug----------------')
    #     # print(train_sample['text'].shape) # [batch x 60]
    #     # print(train_sample['len'].shape) # [batch x 60]
    #     # print(train_sample['labels'].shape) # [batch]
    #     # print(train_sample['sample_indcies'].shape) # [batch]
    #     # print(model(train_sample['text'],train_sample['len']).shape)
    #     # print(train_sample['text'][0]) # [batch x 60]


    #     ''' Training LOOOP'''
    #     accuracy = 0
    #     train_acc_list = []
    #     train_loss_list =[]
    #     dev_acc_list = []
    #     dev_loss_list = []
    #     best_epoch = 0
    #     for epoch in range(args.num_epochs):
    #         print('start epoch %d' % epoch)
    #         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                            sampler=train_sampler,
    #                                            num_workers=args.num_workers,
    #                                            collate_fn=batchify)
    #         log_dict = train(args, model, train_loader, dev_loader, accuracy, device)
    #         if log_dict['new_best']:
    #             best_epoch = epoch

    #         accuracy = log_dict['dev_best_acc']
    #         train_acc_list.append(log_dict['train_acc_epoch'])
    #         train_loss_list.append(log_dict['train_loss_epoch'])
    #         dev_acc_list.append(log_dict['dev_acc_epoch'])
    #         dev_loss_list.append(log_dict['dev_loss_epoch'])
    #         print('----------------------------------------------------------')
    #         print()

    #     # Plotting
    #     train_dict = {'accuracy': np.array(train_acc_list).reshape([-1,]),
    #                     'loss': np.array(train_loss_list).reshape([-1,])}
    #     dev_dict = {'accuracy': np.array(dev_acc_list).reshape([-1]),
    #                 'loss': np.array(dev_loss_list).reshape([-1])}
    #     fig, (ax1, ax2)= plot_model(train_dict, dev_dict, num_epochs=args.num_epochs)
    #     plt.show()
    #     if args.use_glove:
    #         fig.savefig(f'./glove_{args.glove_weights.split(".")[-2]}_plot.png')
    #     else:
    #         fig.savefig(f'./normal_50d_plot.png')

    #     print(f'Best Epoch = {best_epoch}')
    #     print('start testing:\n')
    #     test_dataset = QuestionDataset(test_exs, word2ind, num_classes, class2ind)
    #     test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    #     test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                             batch_size=args.batch_size,
    #                                             sampler=test_sampler,
    #                                             num_workers=args.num_workers,
    #                                             collate_fn=batchify)
    #     test_acc, test_loss = evaluate(test_loader, model, nn.CrossEntropyLoss(), device)
    #     print(f'Test Acc={test_acc}, Test Loss={test_loss}')
