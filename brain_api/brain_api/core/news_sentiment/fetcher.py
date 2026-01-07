"""News fetcher implementations."""

from contextlib import suppress
from datetime import UTC, datetime

import yfinance as yf

from brain_api.core.news_sentiment.models import Article


class YFinanceNewsFetcher:
    """Fetch news articles from yfinance."""

    def fetch(self, symbol: str, max_articles: int) -> list[Article]:
        """Fetch news articles for a symbol using yfinance.

        Args:
            symbol: Stock ticker symbol
            max_articles: Maximum number of articles to fetch

        Returns:
            List of Article objects
        """
        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news or []
        except Exception:
            return []

        articles = []
        for item in news_data[:max_articles]:
            # yfinance news structure changed: data is nested in 'content'
            content = item.get("content", item)  # Fallback to item if no content key

            # Parse published timestamp (now ISO string in pubDate)
            published = None
            pub_date_str = content.get("pubDate")
            if pub_date_str:
                with suppress(ValueError, TypeError):
                    # Parse ISO format: "2025-12-29T21:55:58Z"
                    published = datetime.fromisoformat(
                        pub_date_str.replace("Z", "+00:00")
                    )
            # Fallback to old format (Unix timestamp)
            elif "providerPublishTime" in item:
                with suppress(ValueError, OSError):
                    published = datetime.fromtimestamp(
                        item["providerPublishTime"], tz=UTC
                    )

            # Extract article data from new structure
            # Publisher is now in content.provider.displayName
            provider = content.get("provider", {})
            publisher = (
                provider.get("displayName", "") if isinstance(provider, dict) else ""
            )

            # Link is now in content.canonicalUrl.url or content.clickThroughUrl.url
            link = ""
            canonical_url = content.get("canonicalUrl", {})
            if isinstance(canonical_url, dict) and canonical_url.get("url"):
                link = canonical_url["url"]
            else:
                click_url = content.get("clickThroughUrl", {})
                if isinstance(click_url, dict) and click_url.get("url"):
                    link = click_url["url"]

            # Title and summary
            title = content.get("title", "")
            summary = content.get("summary")

            article = Article(
                title=title,
                publisher=publisher,
                link=link,
                published=published,
                summary=summary,
            )

            # Only include articles with a title
            if article.title:
                articles.append(article)

        return articles


