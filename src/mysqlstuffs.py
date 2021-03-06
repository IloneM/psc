import mysql.connector
from mysql.connector import Error
from six import iteritems,iterkeys,itervalues
import collections

class Database:

    def __init__(self, dbname = None):
        if dbname is None:
            self.dbname = 'psc'
        else:
            self.dbname = dbname
        self.connector = self.connect()
 
    def connect(self):
        """ Connect to MySQL database """
        try:
            conn = mysql.connector.connect(host='localhost',
                                           database=self.dbname,
                                           user='root',
                                           password='cBy1BfHTkPfXUH6ItoYIRBLtgQuvIi4qGtCISjXr')
            if conn.is_connected():
                print('Connected to MySQL database')
                return conn
     
        except Error as e:
            print(e)

    def __get_connector(self):
        if self.connector is not None:
            if not self.connector.is_connected():
                self.connector.reconnect()
        else:
            self.connector = self.connect()
        return self.connector

    def __get_cursor(self):
        return self.__get_connector().cursor()

    def istable(self, table):
        query = "show tables like \'%s\'"
        try:
            cursor = self.__get_cursor()
            cursor.execute(query % (table,))

            return not cursor.fetchone() is None
            #return len(cursor.fetchall()) > 0

        except Error as error:
            print(error)

    def count(self, table, groupby=None):
        query = "SELECT COUNT(*) FROM `%s`"
        if groupby is not None:
            query += ' GROUP BY `%s`' % groupby
        try:
            cursor = self.__get_cursor()
            cursor.execute(query % (table,))

            if groupby is not None:
                return cursor.fetchall()
            else:
                return cursor.fetchone()[0]

        except Error as error:
            print(error)

    def insert(self, table, datas, getids=False):
        if not isinstance(datas, collections.Iterable):
            datas = [datas]

        query = "INSERT INTO `%s`(%s) VALUES(%s)"

        if getids:
            idslist = []
        
        for data in datas:
            assert isinstance(data, dict)

            keystrs = [str(k) for k in iterkeys(data)]
            valstrs = [str(v) for v in itervalues(data)]
            
            try:
                cursor = self.__get_cursor()
                cursor.execute(query % (table, '`' + '`, `'.join(keystrs) + '`', "\'" + "\', \'".join(valstrs) + "\'"))
                
                self.__get_connector().commit()
                
                if getids:
                    cursor.execute("SELECT LAST_INSERT_ID()")
                    tmpret = cursor.fetchone()
                    idslist.append(None if tmpret is None else tmpret[0])

            except Error as error:
                print(error)

        if getids:
            return idslist

    def update(self, table, datas, conds):
        if not isinstance(datas, collections.Iterable):
            datas = [datas]
        if not isinstance(conds, collections.Iterable):
            conds = [conds]

        assert len(datas) == len(conds)

        query = "UPDATE `%s` SET %s WHERE %s"
        
        for data in datas:
            assert isinstance(data, dict)

            datastr = ','.join(['`'+str(k)+"`=\'"+str(v)+"\'" for k,v in iteritems(data)])

        for cond in conds:
            assert isinstance(cond, dict)

            condstr = ','.join(['`'+str(k)+"`=\'"+str(v)+"\'" for k,v in iteritems(cond)])

        try:
            cursor = self.__get_cursor()
            cursor.execute(query % (table, datastr, condstr))
            
            self.__get_connector().commit()

        except Error as error:
            print(error)

    def get(self, table, fieldsid=None, cols=None, idfield='id'):
        if fieldsid is not None and not isinstance(fieldsid, collections.Iterable):
            fieldsid = [fieldsid]

#        if conds is not None and not isinstance(conds, collections.Iterable):
#            conds = [conds]
#        for cond in conds:
#            assert isinstance(cond, dict)
#
#            condstr = ','.join(['`'+str(k)+"`=\'"+str(v)+"\'" for k,v in iteritems(cond)])

        if cols is None:
            cols = '*'
        else:
            if not isinstance(cols, collections.Iterable) or isinstance(cols, str):
                cols = [cols]
            cols = '`' + '`, `'.join(cols) + '`'

        try:
            cursor = self.__get_cursor()
                
            if fieldsid is None or len(fieldsid) <= 0:
                cursor.execute("SELECT %s FROM %s WHERE 1" % (cols, table))
            else:
                cursor.execute("SELECT %s FROM %s WHERE %s IN (%s)" % (cols, '`'+table+'`', idfield, ', '.join([str(i) for i in fieldsid])))

            return cursor.fetchall()

        except Error as error:
            print(error)

    def createtable(self, name, fields):
        assert isinstance(name, str)
        assert isinstance(fields, dict)

        try:
            cursor = self.__get_cursor()
            query = "CREATE TABLE IF NOT EXISTS %s (%s)"
            cursor.execute(query % (name, ', '.join([str(k)+' '+str(v) for k,v in iteritems(fields)])))
        
            self.__get_connector().commit()

        except Error as error:
            print(error)

