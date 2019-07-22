import torch
import torch.nn as nn
import torch.nn.functional as F

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    def __init__(self, vsize, ttoi, embdim, hiddendim):
        super().__init__()
        self.embdim = embdim
        self.hiddendim = hiddendim
        self.vsize = vsize
        self.ttoi = ttoi
        self.tagsize = len(ttoi)

        self.wembed = nn.Embedding(vsize, embdim)
        self.lstm = nn.LSTM(embdim, hiddendim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hiddendim, self.tagsize)

        self.transitions = nn.Parameter(torch.randn(self.tagsize, self.tagsize))
        self.transitions.data[ttoi[START_TAG], :] = -10000
        self.transitions.data[:, ttoi[STOP_TAG]] = -10000

    def _forward_alg(self, feats): #forward the crf
        init_alphas = torch.full((1, self.tagsize), -10000.)
        init_alphas[0][self.ttoi[START_TAG]] = 0.

        forward_var = init_alphas
        for feat in feats: #feats = output of hidden2tag, basically looping per word
            alphas_t = []
            for next_tag in range(self.tagsize): #looping per tag
                emit_score = feat[next_tag].view(1,-1).expand(1, self.tagsize)
                trans_score = self.transitions[next_tag].view(1,-1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1,-1) #row = 1, dont know how many column we want, concat for all the words in sentence
        terminal_var = forward_var + self.transitions[self.ttoi[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence): #get lstm features (emission scores)
        embeds = self.wembed(sentence).view(len(sentence), 1, -1)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hiddendim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags): #calculate the score of true path
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.ttoi[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[self.ttoi[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagsize), -10000.)
        init_vvars[0][self.ttoi[START_TAG]] = 0
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagsize):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1,-1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.ttoi[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop() #pop off/remove the first element/the start
        assert start == self.ttoi[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence): #this is for decoding/prediction/inference
        lstm_feats = self._get_lstm_features(sentence) #get the emission score with bilstm
        score, tag_seq = self._viterbi_decode(lstm_feats) #get the best path with viterbi (prediction)
        return score, tag_seq

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats) #scores all possible sequences
        gold_score = self._score_sentence(feats, tags) #scores of true sequence
        return forward_score - gold_score
