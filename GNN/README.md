# TextING

The code and dataset for the ACL2020 paper [Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks]
(https://arxiv.org/abs/2004.13826), implemented in Tensorflow.

Some functions are based on [Text GCN](https://github.com/yao8839836/text_gcn). Thank for their work.


## Requirements
   

* Python 3.6+
   
* Tensorflow/Tensorflow-gpu 1.12.0
   
* Scipy 1.5.1


## Usage

Download pre-trained word embeddings `glove.6B.300d.txt` from [here](http://nlp.stanford.edu/data/glove.6B.zip) and unzip to the repository.
 

Build graphs from the datasets in `data/corpus/` as:

    python build_graph.py [DATASET] [WINSIZE]

Provided datasets include `mr`,`ohsumed`,`R8`and`R52`. 
 The default sliding window size is 3.

To use your own dataset, put the text file under `data/corpus/` and the label file under `data/` as other datasets do. 
 Preprocess the text by running `remove_words.py` before building the graphs.

Start training and inference as:

    
   python train.py [--dataset DATASET] [--learning_rate LR]
                    
                   [--epochs EPOCHS] [--batch_size BATCHSIZE]
                    
                   [--hidden HIDDEN] [--steps STEPS]
                    
                   [--dropout DROPOUT] [--weight_decay WD]


 To reproduce the result, large hidden size and batch size are suggested as long as your memory allows. We report our result based on 96 hidden size with 1 batch. 
 For the sake of memory efficiency, you may change according to your hardware.


## Citation

    @inproceedings{zhang2020every,
      title={Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks},
      
   author={Zhang, Yufeng and Yu, Xueli and Cui, Zeyu and Wu, Shu and Wen, Zhongzhen and Wang, Liang},
      
   booktitle="Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
      year={2020}
    }



ACL2020论文的代码和数据集[每个文档都拥有其结构：通过图形神经网络进行归纳文本分类]在Tensorflow中实现。
一些功能基于[Text GCN]， 感谢他们的工作。

用法：
下载经过预训练的单词嵌入`glove.6B.300d.txt`，然后解压缩到存储库中。
从`data / corpus /`中的数据集中构建图形为：
     python build_graph.py [DATASET] [WINSIZE]
提供的数据集包括“ mr”，“ ohsumed”，“ R8”和“ R52”。
默认的滑动窗口大小是3。
要使用自己的数据集，请像其他数据集一样，将文本文件放在“ data / corpus /”下，并将标签文件放在“ data /”下。
在构建图形之前，通过运行`remove_words.py`对文本进行预处理。开始训练和推断为：
python train.py [--dataset DATASET] [--learning_rate LR]
                    
                   [--epochs EPOCHS] [--batch_size BATCHSIZE]
                    
                   [--hidden HIDDEN] [--steps STEPS]
                    
                   [--dropout DROPOUT] [--weight_decay WD]


要重现结果，建议您在内存允许的情况下使用较大的隐藏大小和批处理大小。 我们根据1个批次的96个隐藏大小报告结果。为了提高存储效率，可以根据硬件进行更改。
