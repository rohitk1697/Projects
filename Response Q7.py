from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import defaultdict

# Provided responses
responses = """
University A	By the time I ended the game, I was not really looking at the cards. Rather I was looking at the point values and the overall good/bad cards.
University A	You could easily just choose the cards with the best chip status and win very easily. You could speed through it and not really read the cards as the information was not important most of the time.
University A	It seems that there are objectively better paths to take in the game (i.e., one card has more instantaneous benefits with no long-term consequences), which make reading the cards and choosing based off of morality and lore not as important as just choosing the best token-gain card.  
University D	The cards used on the first half, have no effect on the cards used at phase 3.  More cards coul dhave directly impacted future cards.  There became a point where you would only focus on choosing the card that gave the most happiness because the other chips didn't matter as much; as such, I didn't read the card, only the cost.  Similarly, there was no real competition between groups.  Everyone acted independently so you couldn't "screw" over other groups.  It felt like the first half of the game didn't matter because there was a obvious path.
University D	The team colors and steps on the board were too similar in color and made it difficult to differentiate in the beginning.
University A	I feel like the acqusition aspect was overlooked and I only looked at the red, yellow, and blue chip amounts for each card. It does take a very long time to set up. It feels that every team has the same cards until the management cards happen which I feel like isn't the best. 
University A	The rules were a bit confusing, it shifted the focus from learning about acquisition to just focusing on the points
University A	There isn’t a timer so we found some groups took a long time to decide, the game isn’t surprising after the first play through, the period of when to roll the dice isn’t clear in the instructions or game board.
University A	I think the game objective and the goal to educate about acquisitions did conflict slightly- the goal of getting the most user satisfaction points led to a lot of students not reading the cards, and largely picking the cards that gave them the most satisfaction points instead. If the cards were more varied in some of their choices I think students would have to spend more time reading instead of simply choosing the card with the most satisfaction points. 
University A	The colors of the phases and cards match each other which made it confusing to start the game.
University A	I didn't like how some of the cards had types on them such as saying "role" when it should have been "roll". Also the limited options for some of the categories resulted in a lot of us having basically the same setup and the management cards being the only thing that determined who won. 
University A	Too much unstructured time to choose the initial choices (blue section), the red section we were misguided and chose all of the management cards
University A	It was unnecessary to read any part of the card except the specifications at the very top of the card and the token effects at the bottom. The acquisitions itself was less necessary for the game, so I suspect I learned less about acquisitions than intended.
University A	Smaller resolution is better, as having a smaller measurement (more pixels over a given area) means a more detailed image. The resolution for the technologies may have been referring to coverage over an area, but the coverage area was still small. Also, some of the cards seemed imbalanced. For example, the commercial option seemed to have little benefits and the Cost Plus card seemed too beneficial. It was also a little confusing to know when we had to roll the dice.
University C	I wish there was more iconography on the board the pentagon or just related things 
University C	I do wish there was more competition throughout the process of the game
University C	I did not like that most of my decisions were based off the outcome of happiness and not the actual selection 
University C	I didn’t like that the last space involved pulling a card that determined whether or not your product was accepted, but it didn’t have any bearing on who won.
University C	Some aspects of the game did not emphasize the decisions as they pertained to requirements, instead I felt more focused on accumulating chips
University C	The winner of the game is decided at the end with little involvement from other teams.
University C	The game was decided based of tokens and not based of the scenario on the card
University C	I did not like the lack of "consequences". If a project was rejected it should have been an automatic loss or required the team to loop back/start at the begining.
University D	I did not like how it didn't really feel competitive. It is not like you could make things harder for other teams. It did not feel necessary to play with a group because there were no stakes to the game. It just felt like there is a lot of room to add quite a bit to the game to make it more interesting/exciting/competitive. Minor thing, the colors did not match, like the cards to the pieces or the time was red but the chips were orange. There were some misspellings on the card- role instead of roll. 
University C	At first it was complicated to comprehend what’s happened but it was easy to follow along 
University A	Too many cards and categories, sorting out so many piles is both less fun and challenging in constrained playing space.  Rolling dice to play third round was unclear. Once the game is played a single time, the 'most ideal' possibility is easy to see, and can be taken every time which eliminates any discrepancies in choice for the first two phases. 
University D	It’s kind of confusing and jarring. The game takes too long to set up and the process of getting through the first phase is frustrating 
University B	Too complicated to learn on your own and to teach
University B	I did not enjoy the game very much. I think the game should require more teamwork. It was difficult to set up and a little confusing.  It might be better if the game is directly modeled after a popular game like Monopoly.    
University D	I would say the instructions are slightly confusing for first time users.
University D	I did not like how difficult it was to set up the game and get an understanding of what we were doing. In the end, it was dificult to understand how we were learning bout acquisition in the game due to not having a good concept of how to play the game.
University D	It did not feel competitive like a normal game. It felt like we were making decisions with our group then just compared the end results to other groups taking away the competitive nature. 
University B	It was rather confusing starting off the game. Not many of us knew where to start even with participants in the group that had prior experience with the game. 
University B	Starting the game was confusing/difficult
University B	This game was confusing for us to play and U didn’t understand the real strategy behind the game.
University B	It was very confusing and I did not really understand the point of it. 
University B	The amount of cards can somewhat make the game confusing..
"""

# Separating responses by university
responses_by_university = defaultdict(list)
for response in responses.strip().split("\n"):
    university, feedback = response.split("\t")
    responses_by_university[university.strip()].append(feedback.strip())

# Performing sentiment analysis for each university
sentiment_scores = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
for university, feedbacks in responses_by_university.items():
    for feedback in feedbacks:
        analysis = TextBlob(feedback)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            sentiment_scores[university]["positive"] += 1
        elif polarity == 0:
            sentiment_scores[university]["neutral"] += 1
        else:
            sentiment_scores[university]["negative"] += 1

# Ploting sentiment scores
for university, scores in sentiment_scores.items():
    plt.bar(university, scores["positive"], color='g', label='Positive')
    plt.bar(university, scores["neutral"], color='b', label='Neutral', bottom=scores["positive"])
    plt.bar(university, scores["negative"], color='r', label='Negative', bottom=scores["positive"]+scores["neutral"])

plt.xlabel('University')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis by University')
plt.legend(["Positive", "Neutral", "Negative"])
plt.xticks(rotation=45)
plt.show()

