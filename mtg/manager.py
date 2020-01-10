import sqlite3
import mtg


class SQLCommands(object):

    def __init__(self, path):
        self.path = path
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_class, exc, traceback):
        self.conn.commit()
        self.conn.close()

    @classmethod
    def connect(cls, path):
        """
        TODO
        """
        return cls(path)

    @staticmethod
    def execute(cursor, commands, args=[], return_=False):
        """
        Execute sqlite command
        TODO
        """
        if args and not isinstance(args[0], (tuple, list)):
            cursor.execute(commands, args)
        elif args and isinstance(args[0], (tuple, list)):
            cursor.executemany(commands, args)
        else:
            cursor.execute(commands)
        if return_:
            answer = cursor.fetchall()
            return answer

    @staticmethod
    def create(table_name, columns, overwrite=False):
        """
        Args:
            table_name <str> - name of the table
            columns <str> - sqlite columns, e.g. "account TEXT, balance REAL"
        TODO
        """
        table_name = table_name.replace('-', '_')
        columns = columns.replace('-', '_')
        crtt = "CREATE TABLE IF NOT EXISTS {}".format(table_name)
        clmn = "({})".format(columns)
        cmd = " ".join([crtt, clmn])
        return cmd

    @staticmethod
    def update(table_name, column_keys):
        """
        Args:
            table_name <str> - name of the table
            column_keys <tuple(str)> - sqlite columns keys, e.g. ("account", "balance")
        TODO
        """
        table_name = table_name.replace('-', '_')
        column_keys = [c.replace('-', '_') for c in column_keys] \
                      if column_keys else column_keys
        clmn = "{}".format(tuple(column_keys)) if column_keys else ""
        vals = "VALUES ({})".format(",".join(['?' for v in column_keys]))
        cmd = " ".join(["INSERT INTO", table_name, clmn, vals])
        return cmd

    @staticmethod
    def echo(table_name, cursor=None, verbose=True):
        """
        TODO
        """
        echo_cmd = "SELECT * FROM {}".format(table_name)
        if cursor:
            resp = SQLCommands.execute(cursor, echo_cmd, return_=1)
            if verbose:
                print(resp)
        return echo_cmd

    @staticmethod
    def inspect(filename, cursor=None, verbose=True):
        inspect_cmd = "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY 1"
        if cursor:
            resp = SQLCommands.execute(cursor, inspect_cmd, return_=1)
            if verbose:
                print(resp)
        return inspect_cmd
