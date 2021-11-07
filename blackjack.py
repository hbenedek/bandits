#finding optimal betting strategy for the blackjack game using Monte Carlo ES method
import random

class Player():
    def __init__(self) -> None:
        q = None
        policy = None
        returns = None
        cards = 0
        dealer = 0

    def hit(self):
        self.cards += random.randint(1,11)

    def deal(self):
        self.cards = random.randint(1,11) + random.randint(1,11)
        self.dealer = random.randint(1,11)

    def stick(self):
        pass

    def reset(self):
        self.dealer = 0
        self.cards = 0

    def episode(self):
        self.reset()
        self.deal()
        #take action based on policy

#Initialize, for all s ∈ S, a ∈ A(s):
#Q(s, a) ← arbitrary
#π(s) ← arbitrary
#Returns(s, a) ← empty list
#Repeat forever:
#Choose S0 ∈ S and A0 ∈ A(S0) s.t. all pairs have probability > 0
#Generate an episode starting from S0, A0, following π
#For each pair s, a appearing in the episode:
#G ← return following the first occurrence of s, a
#Append G to Returns(s, a)
#Q(s, a) ← average(Returns(s, a))
#For each s in the episode:
#π(s) ← argmaxa Q(s, a)


if __name__=="__main__":
    pass
    