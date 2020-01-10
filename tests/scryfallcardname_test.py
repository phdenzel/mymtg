import pprint
from mtg.scryfall import ScryfallCardName
from tests.prototype import UnitTestPrototype, SequentialTestLoader


class ScryfallCardNameTest(UnitTestPrototype):
    def setUp(self):
        # arguments and keywords
        self.pars = 'atraxa'
        self.kw = {'set': 'c16'}
        # self.kw = {}
        
        self.v = {'verbose': 0}
        # __init__ test
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")
    
    def test_ScryfallCardName1(self):
        """ # ScryfallCardName """
        args = self.pars
        kwargs = self.kw
        self.arg_print(args)
        scry1 = ScryfallCardName(args, format='json', **kwargs)
        pprint.pprint(scry1)
        pprint.pprint(scry1._url)
        pprint.pprint(scry1.data)

    def test_ScryfallCardName2(self):
        """ # ScryfallCardName """
        args = self.pars
        kwargs = self.kw
        self.arg_print(args)
        scry2 = ScryfallCardName(args, format='text', **kwargs)
        pprint.pprint(scry2)
        pprint.pprint(scry2._url)
        pprint.pprint(scry2.data)
        
    def test_ScryfallCardName3(self):
        """ # ScryfallCardName """
        args = self.pars
        kwargs = self.kw
        self.arg_print(args)
        scry3 = ScryfallCardName(args, format='image', **kwargs)
        pprint.pprint(scry3._url)
        pprint.pprint(scry3.data)

    def test_from_txt(self):
        """ # from_txt """
        args = 'data/mtga-hazoret_aggro.txt'
        self.arg_print(args)
        res = ScryfallCardName.from_txt(args)
        for r in res:
            print(r)
            if r:
                pprint.pprint(r[0]._url)
                pprint.pprint(r[0].data)
