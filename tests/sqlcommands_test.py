import sqlite3
import mtg
from mtg.manager import SQLCommands
# from mtg.cards import card_defaults, name2id
import mtg.types.archetypes as atype
from tests.prototype import UnitTestPrototype, SequentialTestLoader

class SQLCommandsTest(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.dbfile = ':memory:'  # 'arkhive.db'
        self.arkhive = mtg.types.tables.ARKHIVE
        self.card_data = card_defaults([None, 1, 'Black Lotus', '2ED',
                                        None, None, '', atype.artifact])
        
        self.v = {'verbose': 0}
        # __init__ test
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_SQLCommands(self):
        """ # SQLCommands """
        args = self.dbfile
        self.arg_print(args)
        db = SQLCommands(args)
        with db as c:
            self.assertIsInstance(db, SQLCommands)
            self.assertIsInstance(c, sqlite3.Cursor)
            print(db)

    def test_connect(self):
        """ # SQLCommands.connect """
        args = self.dbfile
        self.arg_print(args)
        db = SQLCommands.connect(args)
        with db as c:
            self.assertIsInstance(db, SQLCommands)
            self.assertIsInstance(c, sqlite3.Cursor)
            print(db)

    def test_create(self):
        """ # SQLCommands.create """
        args = ('library', self.arkhive)
        self.arg_print(args)
        db = SQLCommands.connect(self.dbfile)
        with db as c:
            create = SQLCommands.create(*args)
            print(create)
            self.assertIsInstance(create, str)

    def test_update(self):
        """ # SQLCommands.update """
        args = ('library', self.arkhive.keys)
        self.arg_print(args)
        db = SQLCommands.connect(self.dbfile)
        with db as c:
            update = SQLCommands.update(*args)
            print(update)
            self.assertIsInstance(update, str)

    def test_echo(self):
        """ # SQLCommands.echo """
        args = 'library'
        self.arg_print(args)
        db = SQLCommands.connect(self.dbfile)
        with db as c:
            echo = SQLCommands.echo(args)
            # echo = SQLCommands.echo('library', c)
            print(echo)
            self.assertIsInstance(echo, str)

    def test_inspect(self):
        """ # SQLCommands.inspect """
        args = self.dbfile
        self.arg_print(args)
        db = SQLCommands.connect(self.dbfile)
        with db as c:
            inspect = SQLCommands.inspect(self.dbfile)
            # inspect = SQLCommands.inspect(self.dbfile, c)
            print(inspect)
            self.assertIsInstance(inspect, str)

    def test_execute(self):
        """ # SQLCommands.execute """
        args = self.card_data
        self.arg_print(args)
        db = SQLCommands.connect(self.dbfile)
        with db as c:
            create = SQLCommands.create('test_table', self.arkhive)
            SQLCommands.execute(c, create)
            update = SQLCommands.update('test_table', self.arkhive.keys)
            SQLCommands.execute(c, update, args=args)
            SQLCommands.echo('test_table', c)


if __name__ == "__main__":

    loader = SequentialTestLoader()
    loader.proto_load(SQLCommandsTest)
    loader.run_suites()
