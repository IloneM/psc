import mysql.connector
from mysql.connector import Error
from six import iteritems,iterkeys,itervalues

class Database:

    dbname = 'psc'

    def __init__(self):
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
        if self.connect is not None:
            return self.connector
        return self.connect()

    def __get_cursor(self):
        return self.__get_connector().cursor()

    def insert(self, table, datas):
        if not isinstance(datas, list):
            datas = [datas]

        query = "INSERT INTO `%s`(%s) VALUES(%s)"
        
        for data in datas:
            assert isinstance(data, dict)

            keystrs = [str(k) for k in iterkeys(data)]
            valstrs = [str(v) for v in itervalues(data)]
            
            try:
                cursor = self.__get_cursor()
                cursor.execute(query % (table, '`' + '`, `'.join(keystrs) + '`', "\'" + "\', \'".join(valstrs) + "\'"))
                
                self.__get_connector().commit()

            except Error as error:
                print(error)

    def update(self, table, datas, conds):
        if not isinstance(datas, list):
            datas = [datas]
        if not isinstance(conds, list):
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

    def get(self, table, fieldsid=None, cols=None):
        if fieldsid is not None and not isinstance(fieldsid, list):
            fieldsid = [fieldsid]

        if cols is None:
            cols = '*'
        else:
            if not isinstance(cols, list) or isinstance(cols, str):
                cols = [cols]
            cols = '`' + '`, `'.join(cols) + '`'

        try:
            cursor = self.__get_cursor()
                
            if fieldsid is None or len(fieldsid) <= 0:
                cursor.execute("SELECT %s FROM %s WHERE 1" % (cols, table))
            else:
                cursor.execute("SELECT %s FROM %s WHERE id IN (%s)" % (cols, '`'+table+'`', ', '.join([str(i) for i in fieldsid])))

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

