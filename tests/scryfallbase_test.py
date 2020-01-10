import pprint
from mtg.scryfall import ScryfallBase
from tests.prototype import UnitTestPrototype, SequentialTestLoader


class ScryfallBaseTest(UnitTestPrototype):
    def setUp(self):
        # arguments and keywords
        self.pars = 'cards/named?fuzzy=atraxa&set=cm2'
        
        self.v = {'verbose': 0}
        # __init__ test
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_ScryfallBase(self):
        """ # ScryfallBase """
        args = self.pars
        self.arg_print(args)
        scry = ScryfallBase(args, format='json')
        pprint.pprint(scry)
        pprint.pprint(scry._url)
        pprint.pprint(scry.raw_data)

    def test_from_txt(self):
        """ # from_txt """
        args = 'data/mtga-hazoret_aggro.txt'
        self.arg_print(args)
        res = ScryfallBase.from_txt(args)
        for r in res:
            print(r)
            if r:
                pprint.pprint(r[0]._url)
                pprint.pprint(r[0].raw_data)
