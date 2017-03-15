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

    def get(self, table, fieldsid=None, rows=None):
        if rows is None:
            rows = '*'
        else:
            rows = ', '.join(rows)

        try:
            cursor = self.__get_cursor()
                
            if fieldsid is None or len(fieldsid) <= 0:
                cursor.execute("SELECT %s FROM %s WHERE 1" % (rows, table))
            else:
                cursor.execute("SELECT %s FROM %s WHERE id IN (%s)" % (rows, '`'+table+'`', ', '.join([str(i) for i in fieldsid])))

            return cursor.fetchall()

        except Error as error:
            print(error)

