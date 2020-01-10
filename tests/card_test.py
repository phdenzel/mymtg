import pprint
from mtg.cards import Card
from mtg.scryfall import ScryfallSearch, ScryfallCardName
from tests.prototype import UnitTestPrototype, SequentialTestLoader


class CardTest(UnitTestPrototype):
    def setUp(self):
        # arguments and keywords
        self.v = {'verbose': 0}
        # __init__ test
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")
    
    def test_Card1(self):
        """ # Card """
        args = ScryfallSearch('set:mh1 name:urza lord')
        self.arg_print(args)
        # pprint.pprint(args.data)
        card1 = Card(**args[0])
        pprint.pprint(card1)
        print(card1.__v__)

    def test_Card2(self):
        """ # Card """
        args = ScryfallSearch('name:Chandra, Torch of Defiance set:KLD cn:110')
        self.arg_print(args)
        # pprint.pprint(args.data)
        card2 = Card(**args[0])
        pprint.pprint(card2)
        print(card2.__v__)

    def test_Card3(self):
        """ # Card """
        args = ScryfallCardName('Black Lotus', set='LEA')
        self.arg_print(args)
        # pprint.pprint(args.data)
        card3 = Card(**args)
        pprint.pprint(card3)
        print(card3.__v__)

    def test_Card4(self):
        """ # Card """
        args = ScryfallCardName('Prismatic Vista', set='MH1')
        self.arg_print(args)
        # pprint.pprint(args.raw_data)
        card4 = Card(**args)
        pprint.pprint(card4)
        print(card4.__v__)

    def test_save_image(self):
        """ # save_image """
        args = ScryfallCardName('Black Lotus')
        self.arg_print(args)
        # pprint.pprint(args.data)
        card4 = Card(**args)
        card4.save_image(version='large')

    def test_from_txt(self):
        """ # from_txt """
        args = 'data/mtga-hazoret_aggro.txt'
        self.arg_print(args)
        deck = Card.from_txt(args)
        for card in deck:
            print(card)
        
        
        
