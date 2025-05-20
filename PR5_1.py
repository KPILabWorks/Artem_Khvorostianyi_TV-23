import feedparser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from urllib.parse import quote

keywords = [
    "climate change", "war", "technology", "artificial intelligence", "economy",
    "health", "education", "sports", "politics", "cybersecurity"
]

def get_google_news(keyword, max_items=30):
    encoded_kw = quote(keyword)
    rss_url = f"https://news.google.com/rss/search?q={encoded_kw}&hl=en&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    entries = feed.entries[:max_items]
    return [{'keyword': keyword, 'title': entry.title, 'summary': entry.summary} for entry in entries]

news_data = []
for kw in keywords:
    news_data.extend(get_google_news(kw))

df = pd.DataFrame(news_data)
df['text'] = df['title'] + ' ' + df['summary']

# Аналіз емоційного тону
df['polarity'] = df['text'].apply(lambda t: TextBlob(t).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda p: 'Positive' if p > 0.1 else ('Negative' if p < -0.1 else 'Neutral'))

# --- ГРАФІКИ ---

# 1. Кількість новин за темами
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='keyword', palette='coolwarm', order=df['keyword'].value_counts().index)
plt.title("Кількість новин за темами")
plt.xlabel("Ключові слова")
plt.ylabel("Кількість")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Настрій по ключовим словам
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='keyword', hue='sentiment', palette='Set2')
plt.title('Емоційний настрій новин по темах')
plt.xlabel('Ключове слово')
plt.ylabel('Кількість новин')
plt.xticks(rotation=45)
plt.legend(title='Настрій')
plt.tight_layout()
plt.show()

# 3. Розподіл Polarities (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='keyword', y='polarity', palette='Pastel2')
plt.title('Розподіл емоційного тону (polarity) по темах')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Word Cloud
text = ' '.join(df['text'].tolist())
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud з усіх новин", fontsize=20)
plt.show()

# 5. Пайчарт розподілу настроїв
plt.figure(figsize=(6, 6))
df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightgray'])
plt.title('Загальний розподіл емоційних настроїв')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 6. Середня полярність по темах
plt.figure(figsize=(10, 6))
avg_polarity = df.groupby('keyword')['polarity'].mean().sort_values()
sns.barplot(x=avg_polarity.values, y=avg_polarity.index, palette='Spectral')
plt.title('Середня емоційна полярність по темах')
plt.xlabel('Середня полярність')
plt.ylabel('Ключове слово')
plt.tight_layout()
plt.show()

# 7. Гістограма полярності з кольорами настроїв
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='polarity', bins=30, kde=True, hue='sentiment', multiple='stack', palette='Set1')
plt.title('Гістограма емоційного тону новин')
plt.xlabel('Полярність')
plt.ylabel('Кількість')
plt.tight_layout()
plt.show()

# 8. Heatmap настроїв по темах
plt.figure(figsize=(10, 6))
sentiment_heatmap = pd.crosstab(df['keyword'], df['sentiment'])
sns.heatmap(sentiment_heatmap, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Heatmap настроїв по ключовим словам')
plt.xlabel('Настрій')
plt.ylabel('Ключове слово')
plt.tight_layout()
plt.show()
