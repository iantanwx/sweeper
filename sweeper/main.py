import asyncio
import json
import os
import re
import traceback
from asyncio import Semaphore
from enum import Enum
from typing import List, Optional
from urllib.parse import urlparse

import aiohttp
import instructor
import tiktoken
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv

# from groq import AsyncGroq
from instructor import Instructor
from openai import AsyncOpenAI
from playwright.async_api import Browser, Playwright, async_playwright
from pydantic import BaseModel, Field, HttpUrl

MAX_APP_PAGE_COUNT = 2

MAX_REVIEWER_PAGE_COUNT = 10

MAX_OPENAI_CONCURRENCY = 2

NUM_STORES_TO_CLASSIFY = 250

NUM_RESULTS_TO_CLASSIFY = 3

# these are the number of tokens per name and message according to OpenAI
# see https://github.com/openai/openai-cookbook/blob/02525b5f3c97919959007cd55982fb6ef3abc788/examples/How_to_count_tokens_with_tiktoken.ipynb
TOKENS_PER_NAME = 1
TOKENS_PER_MESSAGE = 3
MAX_TOKENS_PER_STORE = 10_000 - TOKENS_PER_NAME - TOKENS_PER_MESSAGE

openai_sem = Semaphore(MAX_OPENAI_CONCURRENCY)

visited_stores = set()


class SearchParameters(BaseModel):
    q: str
    type: Optional[str] = None
    autocorrect: Optional[bool] = True
    engine: Optional[str] = None


class Attribute(BaseModel):
    customer_service: Optional[str] = None
    founders: Optional[str] = None
    ceo: Optional[str] = None
    coo: Optional[str] = None
    subsidiaries: Optional[str] = None
    founded: Optional[str] = None
    headquarters: Optional[str] = None


class KnowledgeGraph(BaseModel):
    title: Optional[str] = None
    type: Optional[str] = None
    website: Optional[str] = None
    image_url: Optional[str] = Field(None, alias="imageUrl")
    description: Optional[str] = None
    description_source: Optional[str] = Field(None, alias="descriptionSource")
    description_link: Optional[str] = Field(None, alias="descriptionLink")
    attributes: Optional[Attribute] = None


class Sitelink(BaseModel):
    title: str
    link: str


class Organic(BaseModel):
    title: str
    link: HttpUrl
    position: int
    snippet: Optional[str] = None
    sitelinks: Optional[List[Sitelink]] = None


class TopStory(BaseModel):
    title: Optional[str] = None
    link: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    image_url: Optional[str] = Field(None, alias="imageUrl")


class PeopleAlsoAsk(BaseModel):
    question: Optional[str] = None
    snippet: Optional[str] = None
    title: Optional[str] = None
    link: Optional[str] = None


class RelatedSearch(BaseModel):
    query: str


class SearchResult(BaseModel):
    search_parameters: SearchParameters = Field(..., alias="searchParameters")
    knowledge_graph: Optional[KnowledgeGraph] = Field(None, alias="knowledgeGraph")
    organic: List[Organic] = Field(..., alias="organic")
    top_stories: Optional[List[TopStory]] = Field(None, alias="topStories")
    people_also_ask: Optional[List[PeopleAlsoAsk]] = Field(None, alias="peopleAlsoAsk")
    related_searches: Optional[List[RelatedSearch]] = Field(
        None, alias="relatedSearches"
    )


class ShopifyApp(BaseModel):
    name: str
    description: str
    url: str
    built_for_shopify: bool
    is_appointment_booking_app: bool


prompt_shopify_app_is_appointment_booking_app = """
The following is some HTML describing a Shopify app, including an <a> tag with a link to it.

```html
{html}
```

Your job is to extract the following information:

* name: name of the app
* description: short description of the app
* url: link to the app
* built_for_shopify: a boolean where True means the app has a "Built for Shopify" badge and False means it does not
* is_appointment_booking_app: a boolean where True means the app is an appointment booking app and False means it is not
"""


class BusinessVertical(str, Enum):
    ACCOUNTING = "Accounting"
    ADVERTISING = "Advertising"
    ARCHITECTURE = "Architecture"
    AUTOMOTIVE = "Automotive"
    BUSINESS_CONSULTING = "Business Consulting"
    CATERING = "Catering"
    CLEANING = "Cleaning"
    BEAUTY = "Beauty"
    CONSTRUCTION = "Construction"
    DESIGN = "Design"
    EDUCATION = "Education"
    ENGINEERING = "Engineering"
    ENTERTAINMENT = "Entertainment"
    EVENT_PLANNING = "Event Planning"
    FASHION = "Fashion"
    FINANCIAL_SERVICES = "Financial Services"
    HAIR = "Hair"
    HOME_AND_LIVING = "Home and Living"
    HEALTHCARE = "Healthcare"
    HOSPITALITY = "Hospitality"
    HUMAN_RESOURCES = "Human Resources"
    INFORMATION_TECHNOLOGY = "Information Technology"
    INSURANCE = "Insurance"
    JEWELRY = "Jewelry"
    LANDSCAPING = "Landscaping"
    LEGAL = "Legal"
    LOGISTICS = "Logistics"
    MARKETING = "Marketing"
    PERSONAL_CARE = "Personal Care"
    PHOTOGRAPHY = "Photography"
    PROPERTY_MANAGEMENT = "Property Management"
    PUBLIC_RELATIONS = "Public Relations"
    RETAIL = "Retail"
    REAL_ESTATE = "Real Estate"
    RECRUITMENT = "Recruitment"
    SECURITY = "Security"
    TOURISM = "Tourism"
    TELECOMMUNICATIONS = "Telecommunications"
    TRANSPORTATION = "Transportation"
    TRAVEL = "Travel"
    WASTE_MANAGEMENT = "Waste Management"
    OTHERS = "Others"


class PageClassification(BaseModel):
    contact_information: Optional[str] = None
    vertical: BusinessVertical
    is_shopify: bool
    is_service_probability: int


class Store(BaseModel):
    name: str
    url: str
    contact_information: Optional[str] = None
    vertical: str
    is_shopify: bool


prompt_is_shopify_store = """
Respectively, the following is the contents of the <head> tag, and the <body> of the home page of a potential Shopify store. 

<head> tag:
```html
{head}
```

<body> tag:
```html
{body}
```

Your goal is to determine the following:

* is_shopify: if it is a Shopify store, return True, else False.
* vertical: the vertical of the store. return one of BusinessVertical. If you are not sure, return Others.
* contact_information: a link to the contact page of the store, or the email address of the store if any. If you are not sure, return None.
* is_service_probability: the probability that the store is a service business, expressed as a number between 0 and 100.
"""


async def main():
    load_dotenv()
    async with async_playwright() as p:
        await run(p)


async def run(playwright: Playwright):
    llm = instructor.from_openai(AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
    model = "gpt-4-turbo-preview"
    # llm = instructor.from_groq(AsyncGroq(api_key=os.environ.get("GROQ_API_KEY")))
    # model = "llama3-70b-8192"
    browser = await playwright.chromium.launch()
    page = await browser.new_page()
    await page.goto("https://apps.shopify.com/search?q=appointment+booking")
    await page.wait_for_timeout(5000)

    # collect all review apps
    tasks = []
    page_count = 1
    while True and page_count <= MAX_APP_PAGE_COUNT:
        buttons = page.get_by_role("button")
        for button in await buttons.all():
            attr = await button.get_attribute("data-app-card-target")
            if attr == "wrapper":
                html = await button.inner_html()
                tasks.append(extract_app_info(llm, model, html))
        next_page = page.get_by_role("link", name="Go to Next Page")
        if not await next_page.count():
            break
        await next_page.click()
        await page.wait_for_timeout(5000)
        page_count += 1
    apps = await asyncio.gather(*tasks)

    # for each app, if it is a booking app, visit the page and extract the following from each review:
    # 1. the reviewer
    # 2. the url of the store -- use SERP API to get the url
    reviewers_tasks = []
    for app in apps:
        if app.is_appointment_booking_app:
            reviewers_tasks.append(asyncio.create_task(extract_reviewers(browser, app)))
    reviewers = await asyncio.gather(*reviewers_tasks)
    # Filter out empty lists and flatten the list of reviewers
    reviewers = [reviewer for sublist in reviewers if sublist for reviewer in sublist]

    print(f"Found {len(reviewers)} stores: {reviewers}")

    classifications_tasks = []
    for reviewer in reviewers[:NUM_STORES_TO_CLASSIFY]:
        classifications_tasks.append(search_store(llm, model, reviewer))
    classifications = await asyncio.gather(*classifications_tasks)

    # filter out empty lists and flatten the list of lists
    classifications = [
        classification
        for sublist in classifications
        if sublist
        for classification in sublist
    ]
    print(f"Found {len(classifications)} stores: {classifications}")
    # add the stores to a csv
    with open("stores.csv", "w") as f:
        f.write(
            "name,url,vertical,is_shopify,is_service_probability,contact_information\n"
        )
        for classification in classifications:
            f.write(
                f"{classification.name},{classification.url},{classification.vertical},{classification.is_shopify},{classification.is_service_probability},{classification.contact_information}\n"
            )

    await browser.close()


def get_base_url(url: str) -> str:
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


async def search_store(llm: Instructor, model: str, store_name: str):
    """
    Search for the store on Google and return the results.
    """
    async with aiohttp.ClientSession() as session:
        data = json.dumps(
            {
                "q": f"{store_name} shop",
                "autocorrect": False,
            }
        )
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": os.environ.get("SERPER_API_KEY"),
        }
        async with session.post(
            "https://google.serper.dev/search",
            headers=headers,
            data=data,
        ) as resp:
            resp_json = await resp.json()
            search_result = SearchResult(**resp_json)
            classification_tasks = []
            for result in search_result.organic[:NUM_RESULTS_TO_CLASSIFY]:
                classification_tasks.append(
                    asyncio.create_task(
                        classify_store(
                            llm=llm,
                            model=model,
                            store_name=store_name,
                            store_url=get_base_url(str(result.link)),
                        )
                    )
                )
            classifications = await asyncio.gather(*classification_tasks)
            return [
                classification
                for classification in classifications
                if classification is not None and classification.is_shopify
            ]


def clean_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    head = soup.head
    body = soup.body
    # now strip every unnecessary attribute from <body> to save tokens.
    # only preserve `href` because we want to find contact details, and there may be semantically meaningful links, e.g. to shopify URLs.
    for tag in body.descendants:
        if isinstance(tag, Tag) and tag.attrs:
            for attr in tag.attrs:
                if attr != "href":
                    del tag.attrs[attr]

    return head.str(), body.str()


async def classify_store(llm: Instructor, model: str, store_name: str, store_url: str):
    """
    Classify the store as a Shopify store.
    """
    async with openai_sem:
        try:
            # SERP found a facebook, instagram, or tiktok store, so we don't want to classify it
            if store_url in [
                "https://www.facebook.com",
                "https://www.instagram.com",
                "https://www.tiktok.com",
                "https://www.youtube.com",
                "https://www.amazon.com",
                "https://www.amazon.in",
            ]:
                return None
            if store_url in visited_stores or store_name in visited_stores:
                print(
                    f"Skipping {store_name} at {store_url} because it has already been classified"
                )
                return None
            visited_stores.add(store_url)
            visited_stores.add(store_name)
            print(f"Classifying {store_name} at {store_url}")
            html = await fetch_html(store_url)
            head, body = clean_html(html)
            head_token_count, body_token_count = count_tokens(head), count_tokens(body)
            if head_token_count + body_token_count > MAX_TOKENS_PER_STORE:
                print(
                    f"Truncating {store_name} at {store_url} because its home page has too many tokens"
                )
                encoding = tiktoken.encoding_for_model(model)
                body_tokens = encoding.encode(body)
                max_tokens = MAX_TOKENS_PER_STORE - head_token_count
                body = encoding.decode(body_tokens[:max_tokens])
            classification = await llm.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a web scraper AI.",
                    },
                    {
                        "role": "user",
                        "content": prompt_is_shopify_store.format(head=head, body=body),
                    },
                ],
                model=model,
                response_model=PageClassification,
            )
            return Store(
                name=store_name,
                url=store_url,
                vertical=classification.vertical,
                is_shopify=classification.is_shopify,
                contact_information=classification.contact_information,
            )
        except Exception as e:
            print(
                f"Failed to classify {store_name} at {store_url}: {e}\n{traceback.format_exc()}"
            )
            return None


def is_email(string):
    pattern = r"^(?:mailto:)?[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, string) is not None


def join_url(full_url: str, path_or_full_url: str) -> str:
    # if email return
    if is_email(path_or_full_url):
        return path_or_full_url
    elif path_or_full_url.startswith("http"):
        return path_or_full_url
    elif path_or_full_url.startswith("/"):
        return f"{full_url}{path_or_full_url}"
    else:
        return f"https://{path_or_full_url}"


async def fetch_html(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()


async def extract_reviewers(browser: Browser, app: ShopifyApp):
    page = await browser.new_page()
    await page.goto(app.url)
    await page.wait_for_timeout(5000)

    # if there is a "All Reviews" button, click it and wait
    reviewers = []
    all_reviews = page.get_by_role("link", name="All Reviews")
    if await all_reviews.count():
        await all_reviews.click()
        await page.wait_for_timeout(5000)

        page_count = 1
        while True and page_count <= MAX_REVIEWER_PAGE_COUNT:
            html = await page.content()
            # Pass the HTML content to BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # Extract all divs with the attribute data-merchant-review
            review_divs = soup.find_all("div", attrs={"data-merchant-review": True})
            for div in review_divs:
                # Extract the second child's second child from each review div
                if len(div.contents) >= 4 and len(div.contents[3].contents) >= 1:
                    reviewer = div.contents[3].contents[1].text
                    reviewers.append(reviewer.strip())
            next_page = page.get_by_role("link", name="Go to Next Page")
            if not await next_page.count():
                break
            await next_page.click()
            await page.wait_for_timeout(5000)
            page_count += 1
    await page.close()

    return reviewers


async def extract_app_info(llm: Instructor, model: str, html: str):
    async with openai_sem:
        return await llm.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a web scraper AI.",
                },
                {
                    "role": "user",
                    "content": prompt_shopify_app_is_appointment_booking_app.format(
                        html=html
                    ),
                },
            ],
            model=model,
            response_model=ShopifyApp,
        )


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4-turbo-preview")
    return len(encoding.encode(text)) + TOKENS_PER_NAME + TOKENS_PER_MESSAGE


if __name__ == "__main__":
    asyncio.run(main())
