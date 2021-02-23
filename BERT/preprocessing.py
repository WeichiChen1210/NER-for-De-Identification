"""
Preprocess input data to desired type.
"""
import re

# segmentation
def concat_to_max(sentence_list, max_token_size):
    """Concatenate sub-sentence up to max token size.
    
    First calculate each sentence length of the list.
    Then iterate to combine the sentences if the combined length is smaller than max_token_size

    Parameters:
        sentence_list (Article): list of sentences to be combined
        max_token_size (int): max token size
    
    Return:
        sentences (list): list of combined sentences    
    """
    
    sentence_len = [len(sentence) for sentence in sentence_list]
    sentences = []
    sent_temp = ''
    for idx, (length, sentence) in enumerate(zip(sentence_len, sentence_list)):
        if len(sent_temp) + length <= max_token_size:   # if accumulated sentence's len + current sentence's len <= max
            sent_temp += sentence   # accumulate the current sentence
        else:
            sentences.append(sent_temp) # if > max, append the accumulated sentence
            sent_temp = '' + sentence   # set temp to current sentence
        
        if idx == len(sentence_list) - 1:   # if this is the last one, append it
            sentences.append(sent_temp)
    new_len = [len(sent) for sent in sentences]
    assert sum(new_len) == sum(sentence_len), 'length not matched'
    
    return sentences

def split_to_subsent(sentence, max_token_size):
    """Further split the given sentence to fit the 'max_token_size'
    
    Parameters:
        sentence (str): sentence to be splitted
        max_token_size (int): max token size
    
    Returns:
        sentences (list): list of sentences that are not larger than max_token_size
    """
    # split setence to several subsentences with delimiter period
    if sentence[-1] == '。':
        sentence_split = sentence[:-1].split('。')
        sentence_split = [subsentence + '。' for subsentence in sentence_split]
    else:
        sentence_split = sentence.split('。')
        sentence_split = [subsentence + '。' for subsentence in sentence_split] + [sentence_split[-1]]
    
    # check
    length = [len(item) for item in sentence_split]
    assert sum(length) == len(sentence), 'In \'check_and_split\': the total length of segmented sentences is not equal to original sentence length'
    
    # more checking on the sentence length
    sentences = []
    for subsentence in sentence_split:
        if len(subsentence) > max_token_size:   # if the subsentence is still > max token size
            temp = subsentence.split('，')  # split by camma,
            sub_sent = [sub + '，' if i != (len(temp)-1) else sub for i, sub in enumerate(temp)]  # append camma back except for the last one
            sub_list = concat_to_max(sub_sent, max_token_size) # combine them to bigger sentence whose len < max
            sentences.extend(sub_list)
        else:
            sentences.append(subsentence)
    
    return sentences

def split_sentences(articles, max_token_size=128, max_period_num=4):    # segment 1
    """Segment sentences from articles to fit the max token size limitation.
    
    The function first split articles by counting the accumulated periods up to 'max_period_num'.
    If the segemented sentence length is larger than 'max-token_size', further split it by calling the function 'check_and_split'.

    Parameters:
        articles (list): list of original article contents
        max_token_size (int): the max token size that limits the length of sentences
        max_period_num (int): the max number of periods to segment the articles

    Returns:
        sentences_list (list): list that contains the segmented sentences of each article(like: [[article 1], [article 2], ...])
    """
    sentences_list = []
    for article_id, article in enumerate(articles):
        article_len = len(article)
        article_split = list(article)   # split(tokenize) the article to single words

        sentences = []  # store the cut sentences of this article
        period_count = 0
        start_pos, end_pos = 0, -1

        # search for periods and make a sentence if encounter max num of periods
        for idx, word in enumerate(article_split):            
            if idx == article_len - 1:  # if this is the end of the article, make the collected words to sentences
                end_pos = idx + 1
                sentence = article[start_pos:end_pos]
                
                if (end_pos - start_pos) > max_token_size:
                    sent_split = split_to_subsent(sentence, max_token_size)
                    sentences.extend(sent_split)
                else:
                    sentences.append(sentence)
                continue

            if word == '。':    # encounter a period
                period_count += 1   # record
                if period_count == max_period_num:  # if reach max num of periods
                    end_pos = idx + 1   # end position
                    sentence = article[start_pos:end_pos]   # make a sentence

                    if (end_pos - start_pos) > max_token_size:  # if length of sentence > max token length
                        sent_split = split_to_subsent(sentence, max_token_size)  # further split
                        sentences.extend(sent_split)    # extend the returned list of sentences
                    else:
                        sentences.append(sentence)
                    
                    start_pos = idx + 1
                    end_pos = -1
                    period_count = 0               
        
        start_pos, end_pos, period_count = 0, -1, 0       
        
        # sentences = concat_to_max(sentences, max_token_size)

        # check if the total num of words after segmentation is the same as original article length
        length = [len(sentence) for sentence in sentences]
        if sum(length) != article_len:
            print('In \'sentence_segment\': article', article_id, 'total length not matched')
        
        sentences_list.append(sentences)
        
    return sentences_list

def sub_split(sentence, max_token_size=128):
    """Further split the sentence
    
    Parameters:
        sentence (str): sentence to be further splitted
        max_token_size (int): max length a sentence can reach
    
    Returns:
        result (list): list of sentences where each sentence < max token size    
    """
    # use regex to split the sentence to smaller ones
    pattern = r'([A-Za-z\u4E00-\u9FFF\uFF21-\uFF3A\uFF41-\uFF5A\uff0c\u3001\uff1a\.\d]*[。|？|！|～])'
    split_sent = re.split(pattern, sentence)
    split_sent = [sent for sent in split_sent if len(sent) > 0]
    
    # check if the smaller ones are still longer than max token size
    result = []
    for sent in split_sent:
        if len(sent) > max_token_size:
            sub_sent = sent.split('，')
            sub_sent = [sub + '，' if idx != len(sub_sent) - 1 else sub for idx, sub in enumerate(sub_sent)]
            sub_sent = concat_to_max(sub_sent, max_token_size)
            result.extend(sub_sent)
        else:
            result.append(sent)
    assert sum([len(sub) for sub in result]) == sum([len(sent) for sent in split_sent]), 'In \'sub_split\' the result length does not match the original sentence\'s'
    return result

def split_by_period(articles, max_token_size=128, max_period_num=4):    # segment 2
    """Split the articles into sentences which contains upto 4 periods and max token size
    
    Parameters:
        articles (list): list of articles
        max_token_size (int): max length of sentence
        max_period_num (int): max number of periods in a sentence
    
    Returns:
        result (list): list of sentences where each sentence < max token size and at most max number of perdiods
    """
    result = []
    for article_id, article in enumerate(articles):
        # first use regex to split every sentence with single period
        delim_pos = [match.end() for match in re.finditer('。', article)]
        if delim_pos[-1] != len(article):   # check if the last sentence does not end with period
            delim_pos.append(len(article))
        length = [delim_pos[0]] + [delim_pos[idx] - delim_pos[idx-1] for idx in range(1, len(delim_pos))]   # make a list that records the lengths of the sentences
        assert sum(length) == len(article), 'In \'split_by period\': the sum of the lengths is not equal to the original length'  # check if the sum of the list is the same as the original length
        
        # check if the sentence is shorter than the max token size
        sentences = []
        for idx, l in enumerate(length):
            # make the sentence
            sent = ''
            if idx == 0:
                sent = article[:delim_pos[idx]]
            else:
                sent = article[delim_pos[idx-1]: delim_pos[idx]]            
            # check the size
            if l > max_token_size:   
                sent = sub_split(sent, max_token_size)  # split the sentence to sub-sentences
                sentences.extend(sent)
            else:
                sentences.append(sent)
        assert sum([len(sent) for sent in sentences]) == len(article), 'In \'split_by period\': the total length of the splitted sentences is not equal to the original length' # make sure no words are missed

        # concat sentence upto 4 periods or max token size
        concat_sent = []
        period_count = 0
        temp_str, period = '', '。'
        for idx, sent in enumerate(sentences):
            cur_len = len(sent)
            if period in sent:
                period_count += 1
            
            if idx == len(sentences) - 1:                   # if this is the end of the sentence
                if cur_len + len(temp_str) > max_token_size:    # try to concat, if the length > max token size
                    concat_sent.append(temp_str)                    # append the temp
                    concat_sent.append(sent)                        # append current sentence
                else:                                           # if <= max token size
                    temp_str += sent                                # concat current sentence
                    concat_sent.append(temp_str)                    # append temp str
                temp_str = ''
            elif period_count == max_period_num:                         # else if reach 4 periods
                if cur_len + len(temp_str) > max_token_size:    # try to concat, if the length > max token size
                    concat_sent.append(temp_str)                    # append the temp
                    temp_str = '' + sent                            # update temp str to cur sent
                    period_count = 1                                # update period count
                else:                                           # if length <= max token size
                    temp_str += sent                                # concat the current sentence
                    concat_sent.append(temp_str)                    # append the temp str
                    temp_str = ''                                   # clear temp
                    period_count = 0                                # clear count
            elif cur_len + len(temp_str) > max_token_size:  # else if reach max token size
                concat_sent.append(temp_str)                    # append the temp str
                temp_str = '' + sent                            # update temp to current str
                if period in sent:                              # update the period count according if period is in the sentence
                    period_count = 1
                else:
                    period_count = 0            
            else:                                           # if not reach any stop conditions
                temp_str += sent                            # concat current sentence
        
        assert sum([len(sent) for sent in concat_sent]) == len(article), 'In \'split_by period\': the concat result is not equal to the original length'
        result.append(concat_sent)
    
    return result

def split_by_regex(articles, max_token_size=128):   # segment 3
    """Split the sentences by regular expression

    Parameters:
        articles (list): list of articles
        max_token_size (int): max sentence length
    
    Return:
        split_articles (list): list of segmented aticlesd    
    """
    # (all chinese words)：(all chinese words, English words, '，', '、', '。', '？', '…', '！', '～', '.', digits)
    pattern = r'([\u4E00-\u9FFF]*：[A-Za-z\u4E00-\u9FFF\uff0c\u3001\u3002\uff1f\u2026\uff01\uff5e\.\d]*[。|？|！|…|～])'
    
    # split every article into sentences
    split_articles = []
    for idx, article in enumerate(articles):
        split_art = re.split(pattern, article)
        split_art = [piece for piece in split_art if len(piece) > 0]  # clean empty items
        length = [len(sent) for sent in split_art]
        assert sum(length) == len(article), 'article {} regex result not matched'.format(idx)
        
        # check sentence length
        split_sent = []
        for idx, leng in enumerate(length):
            if leng > max_token_size:   # if > max, split
                sub_sent = sub_split(split_art[idx], max_token_size)
                split_sent.extend(sub_sent)
            else:
                split_sent.append(split_art[idx])    
        assert sum([len(sent) for sent in split_sent]) == len(article), 'article {} split result not matched'.format(idx)

        # concatenate sentences up to max length
        split_sent = concat_to_max(split_sent, max_token_size) 
        
        split_articles.append(split_sent)
    
    return split_articles

def concat_to_max_overlap(sentence_list, overlap_len, max_token_size):
    """Concatenate sub-sentence up to max token size.
    
    First calculate each sentence length of the list.
    Then iterate to combine the sentences if the combined length is smaller than max_token_size

    Parameters:
        sentence_list (Article): list of sentences to be combined
        overlap_len (int) : overlapped subsentence's len / 0 if no overlapped subsentence
        max_token_size (int): max token size
    
    Return:
        sentences (list): list of combined sentences
        ovl_length (list): list of overlapped subsentences's len
    """
    
    sentence_len = [len(sentence) for sentence in sentence_list]
    sentences = []
    ovl_length = []
    sent_temp = ''
    sent_pre = ''
    sent_pre_len = 0
    for idx, (length, sentence, ol_len) in enumerate(zip(sentence_len, sentence_list, overlap_len)):
        if ol_len != 0: #check if overlap happened
            sentences.append(sentence)
            ovl_length.append(ol_len)
        elif len(sent_temp) + length <= max_token_size:   # if accumulated sentence's len + current sentence's len <= max
            sent_temp += sentence   # accumulate the current sentence
            sent_pre = sentence
            sent_pre_len = len(sent_pre)
        else:
            sentences.append(sent_temp) # if > max, append the accumulated sentence
            ovl_length.append(sent_pre_len)
            
            if sent_pre_len + len(sentence) > max_token_size: # check if new sentences;s len > max
                sentences.append(sent_pre)
                ovl_length.append(0)
                sent_pre = ''
                
            sent_temp = sent_pre + sentence   # set temp to current sentence
            sent_pre = sentence
            sent_pre_len = len(sent_pre)
        
        if idx == len(sentence_list) - 1:   # if this is the last one, append it
            sentences.append(sent_temp)
            ovl_length.append(ol_len)
    new_len = [len(sent) for sent in sentences]
    #assert sum(new_len) == sum(sentence_len), 'length not matched'
    
    return sentences, ovl_length

def sub_split_overlap(sentence, max_token_size=128):
    """Further split the sentence
    
    Parameters:
        sentence (str): sentence to be further splitted
        max_token_size (int): max length a sentence can reach
    
    Returns:
        result (list): list of sentences where each sentence < max token size 
        reslen (list): list of overlapped subsentences' len
    """
    # use regex to split the sentence to smaller ones
    pattern = r'([A-Za-z\u4E00-\u9FFF\uFF21-\uFF3A\uFF41-\uFF5A\uff0c\u3001\uff1a\.\d]*[。|？|！|…|～|．])'
    split_sent = re.split(pattern, sentence)
    split_sent = [sent for sent in split_sent if len(sent) > 0]
    
    # check if the smaller ones are still longer than max token size
    result = []
    reslen = []
    for sent in split_sent:
        if len(sent) > max_token_size:
            sub_sent = sent.split('，')
            sub_sent = [sub + '，' if idx != len(sub_sent) - 1 else sub for idx, sub in enumerate(sub_sent)]
            sub_len = [0 for i in range(len(sub_sent))]
            sub_sent, sub_len = concat_to_max_overlap(sub_sent, sub_len, max_token_size)
            result.extend(sub_sent)
            reslen.extend(sub_len)
        else:
            result.append(sent)
            reslen.append(len(sent))

    #assert sum([len(sub) for sub in result]) == sum([len(sent) for sent in split_sent]), 'In \'sub_split\' the result length does not match the original sentence\'s'
    return result, reslen

def split_by_regex_overlap(articles, max_token_size=128):   # segment 4
    """Split the sentences by regular expression

    Parameters:
        articles (list): list of articles
        max_token_size (int): max sentence length
    
    Return:
        split_articles (list): list of segmented aticlesd    
    """
    # (all chinese words)：(all chinese words, English words, '，', '、', '。', '？', '…', '！', '～', '.', digits)
    pattern = r'([\u4E00-\u9FFF]*：[A-Za-z\u4E00-\u9FFF\uff0c\u3001\u3002\uff1f\u2026\uff01\uff5e\.\d]*[。|？|！|…|～|．])'
    
    # split every article into sentences
    split_articles = []
    sentence_len = []
    for idx, article in enumerate(articles):
        split_art = re.split(pattern, article)
        split_art = [piece for piece in split_art if len(piece) > 0]  # clean empty items
        length = [len(sent) for sent in split_art]
        #assert sum(length) == len(article), 'article {} regex result not matched'.format(idx)
        
        # check sentence length
        split_sent = []
        split_len = []
        for idx, leng in enumerate(length):
            if leng > max_token_size:   # if > max, split
                sub_sent, sub_len = sub_split_overlap(split_art[idx], max_token_size)
                split_sent.extend(sub_sent)
                split_len.extend(sub_len)
            else:
                split_sent.append(split_art[idx])
                split_len.append(0)
       # assert sum([len(sent) for sent in split_sent]) == len(article), 'article {} split result not matched'.format(idx)

        # concatenate sentences up to max length
        split_sent, split_len = concat_to_max_overlap(split_sent, split_len, max_token_size) 
        
        split_articles.append(split_sent)
        sentence_len.append(split_len)
    
    return split_articles, sentence_len