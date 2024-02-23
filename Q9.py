import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Responses data
responses = [
    {"university": "University A", "comment": "Please elaborate on the rules you found confusing."},
    {"university": "University A", "comment": "It was not clear. Some of the title did not match with the cards"},
    {"university": "University A", "comment": "The instruction booklet didn't really describe how for Phases 1 and 2 you go through each individual step before rolling the dice. Also the selecting of the technical enhancements and the overall requirements could have been laid out in an easier to understand format."},
    {"university": "University A", "comment": "When we were supposed to roll the dice, how many technical enhancements you could/have to choose, when/which cards we look at"},
    {"university": "University A", "comment": "There was a unit called 'sq/m.' The closest unit to this would probably be meters squared; they would make more sense in terms of square mile. However, for resolution, the requirements were 'at least 10 sq/m,' which does not make sense, as the most statement from this is that each pixel encodes at least 10 square miles, and a smaller area per pixel results in a better image. The number of the technical solution was also given the units (100 sq/m), which I took to mean 100 units per square mile, but this is not what the statement would actually say. The rules regarding gameplay were much more clear."},
    {"university": "University C", "comment": "It did not seem very clear what the objective was, besides to win the most chips. It was also confusing if when we landed on the had to stop wild card spot if we continued moving or continue our turn."},
    {"university": "University C", "comment": "The trading coins"},
    {"university": "University A", "comment": "Selecting enhancements. Typos on enhancement cards and directions page."},
    {"university": "University D", "comment": "The beginning part of the game. I had already played so I knew how to set up the game and everything but my other classmates found it confusing"},
    {"university": "University B", "comment": "How to start the game, moving around the board, and why not just try to gather all the yellow chips"},
    {"university": "University B", "comment": "I think the most confusing part was the dice, when to roll or when to move."},
    {"university": "University B", "comment": "The book was confusing with how you should start the game and what type of cards you will need."},
    {"university": "University D", "comment": "How to start the game in a sense of picking the technical solutions and their enhancements. Also reading all of the requirements for the technical solutions."},
    {"university": "University D", "comment": "It was hard trying to understand what the rules were stating when it came to starting the game and how the progress of the game was supposed to go."},
    {"university": "University B", "comment": "The book is rather informative but in my opinion could be more clear. Like me and my fellow peers found ourselves repeatedly reading the instructions over again to try to get a gist of how to start the game."},
    {"university": "University B", "comment": "It seemed thrown together where it didn't need to be played as a board game"},
    {"university": "University B", "comment": "All the giving and taking of the pieces."}
]

# Creating a DataFrame
df = pd.DataFrame(responses)

# Sentiment Analysis
df['sentiment'] = df['comment'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Plotting Sentiment Analysis
plt.figure(figsize=(10, 6))
plt.bar(df['university'], df['sentiment'], color='skyblue')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
plt.title('Sentiment Analysis of Feedback Comments')
plt.xlabel('University')
plt.ylabel('Sentiment Polarity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
