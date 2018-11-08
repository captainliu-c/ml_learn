在生成batch期间[make_data函数中]，会生成x_sub、y_sub、sentence_lengths3个中间文件。3个中间文件是处理完毕的原始文件，等待拆分成batch。

在生成batch期间[make_data函数中]，会生成vocab_dict，方便最后公布提交数据时，可以通过相同的词典来映射。

在预测submit数据时[train.py->main函数中]，会生成按照预测完毕，并且添加句号的中间文件files，等待获得实体类别和索引情况后写入到对应的ann中。