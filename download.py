
import os
import pandas as pd
from queue import Queue
import requests
from threading import Thread
from tqdm import tqdm

class Downloader(Thread):
    def __init__(self, q, url, *args, **kwargs):
        self.q = q
        self.url = url
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            content = requests.get(self.url).content.decode()
        except Exception as e:
            content = str(e)

        self.q.put((self.url, content))


def get_iterator(seq, verbose=False):
    if verbose:
        iterator = tqdm(seq)
    else:
        iterator = iter(seq)

    return iterator

def get_pages(links, CHUNK_SIZE=50, verbose=True):
    q = Queue()
    chunks = 0
    total_chunks = (len(links) // CHUNK_SIZE) + 1
    results = {}

    iterator = get_iterator(range(total_chunks), verbose)

    for i in iterator:
        ls = links[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        threads = [Downloader(q, url) for url in ls]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        while not q.empty():
            url, content = q.get()
            results[url] = content

    return results

def get_links(source, endpoint, CHUNK_SIZE=50, verbose=True):
    first_page = requests.get(source+endpoint).content.decode()
    soup = bs(first_page, 'html.parser')
    num_pages = int(soup.find('h7').get_text().split()[-1])

    urls = [source+endpoint+"&p={}".format(i+1) for i in range(num_pages)]

    pages = get_pages(urls, CHUNK_SIZE=CHUNK_SIZE, verbose=verbose)
    links = []

    iterator = get_iterator(pages.values(), verbose)

    for page in iterator:
        soup = bs(page, 'html.parser')
        rows = soup.find('table').find_all('tr')

        for row in rows:
            cols = row.find_all('td')

            if len(cols):
                link = source + row.find('a').attrs['href']

                if "TRUMP" in cols[-1].get_text().upper() or \
                   'tweet' in link:
                   pass
                else:
                    links.append(link)

    return links


if __name__ == '__main__':
    this_desc = 'Download official statement examples from Trump '
                'administration officials.'
    parser = argparse.ArgumentParser(description=this_desc)
    parser.add_argument('-d', '--data', required=True,
                        help='The extracted source code snippets for encoding.')
    parser.add_argument('-D', '--dir', default=os.getcwd(),
                        help='Number of concurrent threads when downloading links.')
    parser.add_argument('-u', '--url', default="https://justfacts.votesmart.org", 
                        help='VoteSmart source URL.')
    parser.add_argument('-e', '--endpoint', 
                        default="/public-statements/NA/P/?s=date&section=officials", 
                        help='Endpoint for downloading public statement records.')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Report processing progress')
    parser.add_argument('-c', '--chunk_size', default=50, type=int,
                        help='Number of concurrent threads when downloading links.')

    args = parser.parse_args()

    links = get_links(args.url, args.endpoint, args.chunk_size, args.verbose)
    texts = get_pages(links, args.chunk_size, args.verbose)
    iterator = get_iterator(texts, args.verbose)
    rows = {}

    for url in iterator:
        soup = bs(texts[url], 'html.parser')
        text = soup.find("div", {"id": "publicStatementDetailSpeechContent"})

        if text is None:
            if args.verbose:
                print("Statement text missing: " + url)
            continue
        else:
            text = text.get_text()

        headlines = soup.find('div', {'class': 'row mt-4'})

        if headlines is None:
            if args.verbose:
                print("Header info missing: " + url)
            continue
        else:
            headlines = headlines.find_all('div', {'class': 'col'})

        official, stmt_date, _, issue = [x.get_text().split(":")[-1] for x in headlines]
        rows[url] = {'text': text.strip(), 
                     'official': official.strip(), 
                     'stmt_date': stmt_date.strip(), 
                     'issue': issue.strip()}

    df = pd.DataFrame.from_dict(rows, orient='index')
    df = df.reset_index()
    df = df.rename({'index': 'link'}, axis=1)
    df['stmt_date'] = pd.to_datetime(df.stmt_date)

    df.to_csv(os.path.join(args.dir, "statements.csv"), index=False)

    if args.verbose:
        print("Done! CSV saved at: {}".format(os.path.join(args.dir, "statements.csv")))
