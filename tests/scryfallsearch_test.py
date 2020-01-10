import pprint
from mtg.scryfall import ScryfallSearch
from tests.prototype import UnitTestPrototype, SequentialTestLoader


class ScryfallSearchTest(UnitTestPrototype):
    def setUp(self):
        # arguments and keywords
        self.pars = ''
        self.kw = {}
        
        self.v = {'verbose': 0}
        # __init__ test
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")
    
    def test_ScryfallSearch1(self):
        """ # ScryfallSearch """
        args = 'set:c16 name:atraxa'
        kwargs = self.kw
        self.arg_print(args)
        scry1 = ScryfallSearch(args, format='json', **kwargs)
        pprint.pprint(scry1)
        pprint.pprint(scry1._url)
        pprint.pprint(scry1.data)

    def test_from_txt(self):
        """ # from_txt """
        args = 'data/mtga-hazoret_aggro.txt'
        self.arg_print(args)
        res = ScryfallSearch.from_txt(args)
        for r in res:
            print(r)
            if r:
                pprint.pprint(r[0]._url)
                pprint.pprint(r[0].data)
