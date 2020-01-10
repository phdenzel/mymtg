import pprint
from mtg.scryfall import ScryfallSets
from tests.prototype import UnitTestPrototype, SequentialTestLoader


class ScryfallSetsTest(UnitTestPrototype):
    def setUp(self):
        # arguments and keywords
        self.kw = {}

        self.v = {'verbose': 0}
        # __init__ test
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")
    
    def test_ScryfallSets1(self):
        """ # ScryfallSets """
        args = ''
        kwargs = self.kw
        self.arg_print(args)
        scry1 = ScryfallSets(args, **kwargs)
        pprint.pprint(scry1)
        pprint.pprint(scry1._url)
        pprint.pprint(scry1.data)

    def test_ScryfallSets2(self):
        """ # ScryfallSets """
        args = 'mh1'
        kwargs = self.kw
        self.arg_print(args)
        scry2 = ScryfallSets(args, **kwargs)
        pprint.pprint(scry2)
        pprint.pprint(scry2._url)
        pprint.pprint(scry2.data)

    def test_all_sets(self):
        """ # all_sets """
        args = ''
        self.arg_print(args)
        sets = ScryfallSets.all_sets()
        pprint.pprint(sets)
