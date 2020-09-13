import os
import re
import aiohttp
import asyncio
import urllib
import parse
import pprint


class ScryfallBase(object):
    """
    Scry X
    """
    base_url = 'https://api.scryfall.com'
    url_key = ''

    def __init__(self, query, _url=None, **kwargs):
        """
        Args:
            query <str> - raw query (url string formatted)

        Kwargs:
            _url <str> - if not None, complete url will be overidden
            format <str> - *json*|text|image
            face <str> - returns backside of image if there is one for face == 'back'
            version <str> - small|normal|*large*|png|art_crop|border_crop
            pretty <bool> - *False*, otherwise json is prettified
        """
        if not hasattr(self, 'pars'): self.pars = {}
        self.pars.update({'format': kwargs.get('format', 'json'),
                          'face': kwargs.get('face', ''),
                          'version': kwargs.get('version', ''),
                          'pretty': kwargs.get('pretty', '')})
        self.enc_pars = urllib.parse.urlencode(self.pars)
        self._url = '{0}/{1}/{2}&{3}'.format(self.base_url, self.url_key,
                                             query, self.enc_pars)
        if _url:
            self._url = _url

        async def getRequest(client, url, **kwargs):
            async with client.get(url, **kwargs) as response:
                if self.pars['format'] == 'json':
                    return await response.json()
                if self.pars['format'] == 'text':
                    return await response.text()
                elif self.pars['format'] == 'image':
                    return await response.read()

        async def main(loop):
            async with aiohttp.ClientSession(loop=loop) as client:
                self.raw_data = await getRequest(client, self._url)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(loop))

        # if isinstance(self.data, bytes):
        #     with open('cache/image.jpg','wb') as f:
        #         f.write(self.data)

    def keys(self):
        if hasattr(self, 'data'):
            return self.data.keys()
        elif hasattr(self, 'raw_data'):
            return self.raw_data.keys()

    def __getitem__(self, key):
        if hasattr(self, 'data'):
            return self.data[key]
        elif hasattr(self, 'raw_data'):
            return self.raw_data[key]

    @classmethod
    def save_image(cls, query, image_name='cache/image.jpg', **kwargs):
        scry = cls(query, format='image', **kwargs)
        if os.path.exists(image_name):
            image_name = image_name.replace('.jpg', '_1.jpg')
        if isinstance(scry.raw_data, bytes):
            with open(image_name,'wb') as f:
                f.write(scry.raw_data)
        else:
            print("No image data found!")
        
    @classmethod
    def from_txt(cls, txtfile, url_quote=True,
                 line_format="{amount:d} {name} ({set}) {collector_number:d}",
                 arg_search='cards/named?fuzzy={name:}&set={set:}',
                 kwarg_search={}, **kwargs):
        """
        Scry cards from text file; default line format: <[amount] [name] ([set]) [collector_number]>
        """
        if isinstance(txtfile, str):
            with open(txtfile, 'r') as f:
                txtdata = f.readlines()
        else:
            txtdata = txtfile.readlines()
        for line in txtdata:
            line = line.strip()
            parsed = parse.parse(line_format, line)
            if parsed:
                params = parsed.named
                N = params.pop('amount', 1)
                # args
                if url_quote:
                    args = arg_search.format(**{k: urllib.parse.quote(params[k])
                                                for k in params if k in arg_search})
                else:
                    args = arg_search.format(**{k: params[k] for k in params
                                                if k in arg_search})
                # excuses
                if params.get('set', 'NON') == 'NON' or params.get('collector_number', 0) < 1:
                    args = re.findall("(.+)set", args)[0].strip() if 'set' in args else args
                    params['set'] = ''
                # keywords
                kw = kwargs
                for k in kwarg_search:
                    if k in params:
                        kw[k] = params[k]
                # scry
                yield N*[cls(args, **kw)]
            else:
                yield None


class ScryfallCardName(ScryfallBase):
    """
    Scry card names (in different sets)
    """
    url_key = 'cards'

    def __init__(self, name, fuzzy=True, **kwargs):
        """
        Args:
            name <str> - Approximate name (has to be unique enough tough)

        Kwargs:
            fuzzy <bool> - perform fuzzy search if True, otherwise exact
            set <str> - set code to clear up ambiguities
            _url <str> - if not None, complete url will be overidden
            format <str> - *json*|text|image
            face <str> - returns backside of image if there is one for face == 'back'
            version <str> - small|normal|*large*|png|art_crop|border_crop
            pretty <bool> - *False*, otherwise json is prettified
        """
        self.pars = kwargs
        if fuzzy:
            query = 'named?fuzzy={}'.format(urllib.parse.quote(name))
        else:
            query = 'named?exact={}'.format(urllib.parse.quote(name))
        super(ScryfallCardName, self).__init__(query, **kwargs)
        if isinstance(self.raw_data, dict) and 'data' in self.raw_data:
            self.data = self.raw_data['data']
        else:
            self.data = self.raw_data

    @classmethod
    def from_txt(cls, txtfile, url_quote=False,
                 arg_search='{name:}', kwarg_search={'set': ''}, **kwargs):
        """
        Scry cards from text file; default line format: <[amount] [name] ([set]) [collector_number]>
        """
        kwargs.setdefault('fuzzy', True)
        return super(ScryfallCardName, cls).from_txt(
            txtfile, url_quote=url_quote,
            arg_search=arg_search, kwarg_search=kwarg_search, **kwargs)


class ScryfallSearch(ScryfallBase):
    """
    Scry ?
    """
    url_key = 'cards'

    def __init__(self, text='',
                 unique='cards', order='', dir='auto', **kwargs):
        """
        Args:
            text <str> - search text (formatted as in https://scryfall.com/docs/syntax)

        Kwargs:
            unique <str> - handling of duplicates: *cards*|art|prints
            order <str> - ordering scheme: name|set|*released*|rarity|color|cmc||power|toughness
            dir <str> - direction of ordering: *auto*|asc|desc
            _url <str> - if not None, complete url will be overidden
            format <str> - *json*|csv
            pretty <bool> - *False*, otherwise json is prettified
        """
        q = urllib.parse.quote(str(text))
        query = 'search?unique={0}&order={1}&dir={2}&q={3}'.format(unique, order, dir, q)
        super(ScryfallSearch, self).__init__(query, **kwargs)
        if isinstance(self.raw_data, dict) and 'data' in self.raw_data:
            self.data = self.raw_data['data']
        else:
            self.data = self.raw_data

    @classmethod
    def from_txt(cls, txtfile, url_quote=False,
                 arg_search='name:{name:} set:{set:}', **kwargs):
        """
        Scry cards from text file; default line format: <[amount] [name] ([set]) [collector_number]>
        """
        return super(ScryfallSearch, cls).from_txt(
            txtfile, url_quote=url_quote, arg_search=arg_search, **kwargs)


class ScryfallSets(ScryfallBase):
    """
    Scry sets
    """
    url_key = 'sets'

    def __init__(self, code='', fuzzy=True, **kwargs):
        """
        Args:
            code <str> - set code

        Kwargs:
            _url <str> - if not None, complete url will be overidden
            pretty <bool> - *False*, otherwise json is prettified
        """
        kwargs['format'] = 'json'
        query = '{}?'.format(code)
        super(ScryfallSets, self).__init__(query, **kwargs)
        if isinstance(self.raw_data, dict) and 'data' in self.raw_data:
            self.data = self.raw_data['data']
        else:
            self.data = self.raw_data

    @staticmethod
    def all_sets():
        """
        Retrieve all set codes and names

        Return:
            sets <dict> - set data as dictionary
        """
        scry = ScryfallSets()
        sets = {}
        for e in scry.data:
            sets[e['code']] = e['name']
        return sets
        
