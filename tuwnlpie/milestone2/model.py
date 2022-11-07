import torch
import torch.nn.functional as F
from torch import nn


class BoWClassifier(nn.Module):  # inheriting from nn.Module!
    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.
        # Torch defines nn.Linear(), which provides the affine map.
        # Note that we could add more Linear Layers here connected to each other
        # Then we would also need to have a HIDDEN_SIZE hyperparameter as an input to our model
        # Then, with activation functions between them (e.g. RELU) we could have a "Deep" model
        # This is just an example for a shallow network
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec, sequence_lens):
        # Ignore sequence_lens for now!
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        # Softmax will provide a probability distribution among the classes
        # We can then use this for our loss function
        return F.log_softmax(self.linear(bow_vec), dim=1)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
