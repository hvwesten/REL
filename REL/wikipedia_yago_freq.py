import os, sqlite3, json
import re
from urllib.parse import unquote

import numpy as np

from REL.db.generic import GenericLookup
from REL.utils import first_letter_to_uppercase, trim1, unicode2ascii

"""
Class responsible for processing Wikipedia dumps. Performs computations to obtain the p(e|m) index and counts 
overall occurrences of mentions.
"""


class WikipediaYagoFreq:
    def __init__(self, base_url, wiki_version, wikipedia):
        self.base_url = base_url
        self.wiki_version = wiki_version
        self.wikipedia = wikipedia

        self.wiki_freq = {}
        self.p_e_m = {}
        self.mention_freq = {}

    def store(self):
        """
        Stores results in a sqlite3 database.

        :return:
        """
        print("Please take a break, this will take a while :).")

        wiki_db = GenericLookup(
            "entity_word_embedding",
            os.path.join(self.base_url, self.wiki_version, "generated"),
            table_name="wiki",
            columns={"p_e_m": "blob", "lower": "text", "freq": "INTEGER"},
        )

        wiki_db.load_wiki(self.p_e_m, self.mention_freq, batch_size=50000, reset=True)

    def compute_wiki(self):
        """
        Computes p(e|m) index for a given wiki and crosswikis dump.

        :return:
        """

        self.__wiki_counts()
        self.__cross_wiki_counts()

        # Step 1: Calculate p(e|m) for wiki.
        print("Filtering candidates and calculating p(e|m) values for Wikipedia.")
        for ent_mention in self.wiki_freq:
            if len(ent_mention) < 1:
                continue

            ent_wiki_names = sorted(
                self.wiki_freq[ent_mention].items(), key=lambda kv: kv[1], reverse=True
            )
            # Get the sum of at most 100 candidates, but less if less are available.
            total_count = np.sum([v for k, v in ent_wiki_names][:100])

            if total_count < 1:
                continue

            self.p_e_m[ent_mention] = {}

            for ent_name, count in ent_wiki_names:
                self.p_e_m[ent_mention][ent_name] = count / total_count

                if len(self.p_e_m[ent_mention]) >= 100:
                    break

        del self.wiki_freq

    def compute_custom(self, custom=None):
        """
        Computes p(e|m) index for YAGO and combines this index with the Wikipedia p(e|m) index as reported
        by Ganea et al. in 'Deep Joint Entity Disambiguation with Local Neural Attention'.

        Alternatively, users may specificy their own custom p(e|m) by providing mention/entity counts.


        :return:
        """
        if custom:
            self.custom_freq = custom
        else:
            self.custom_freq = self.__yago_counts()

        print("Computing p(e|m)")
        for mention in self.custom_freq:
            total = len(self.custom_freq[mention])

            # Assumes uniform distribution, else total will need to be adjusted.
            if mention not in self.mention_freq:
                self.mention_freq[mention] = 0
            self.mention_freq[mention] += 1
            cust_ment_ent_temp = {
                k: 1 / total for k, v in self.custom_freq[mention].items()
            }

            if mention not in self.p_e_m:
                self.p_e_m[mention] = cust_ment_ent_temp
            else:
                for ent_wiki_id in cust_ment_ent_temp:
                    prob = cust_ment_ent_temp[ent_wiki_id]
                    if ent_wiki_id not in self.p_e_m[mention]:
                        self.p_e_m[mention][ent_wiki_id] = 0.0

                    # Assumes addition of p(e|m) as described by authors.
                    self.p_e_m[mention][ent_wiki_id] = np.round(
                        min(1.0, self.p_e_m[mention][ent_wiki_id] + prob), 3
                    )

    def __yago_counts(self):
        """
        Counts mention/entity occurrences for YAGO.

        :return: frequency index
        """

        num_lines = 0
        print("Calculating Yago occurrences")
        custom_freq = {}
        with open(
            os.path.join(self.base_url, "generic/p_e_m_data/aida_means.tsv"),
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                num_lines += 1

                if num_lines % 5000000 == 0:
                    print("Processed {} lines.".format(num_lines))

                line = line.rstrip()
                line = unquote(line)
                parts = line.split("\t")
                mention = parts[0][1:-1].strip()

                ent_name = parts[1].strip()
                ent_name = ent_name.replace("&amp;", "&")
                ent_name = ent_name.replace("&quot;", '"')

                x = ent_name.find("\\u")
                while x != -1:
                    code = ent_name[x : x + 6]
                    replace = unicode2ascii(code)
                    if replace == "%":
                        replace = "%%"

                    ent_name = ent_name.replace(code, replace)
                    x = ent_name.find("\\u")

                ent_name = self.wikipedia.preprocess_ent_name(ent_name)
                if ent_name in self.wikipedia.wiki_id_name_map["ent_name_to_id"]:
                    if mention not in custom_freq:
                        custom_freq[mention] = {}
                    ent_name = ent_name.replace(" ", "_")
                    if ent_name not in custom_freq[mention]:
                        custom_freq[mention][ent_name] = 1

        return custom_freq

    def __cross_wiki_counts(self, using_database=False):
        """
        Updates mention/entity for Wiki with this additional corpus.

        :return:
        """

        print("Updating counts by merging with CrossWiki")

        # c = None
        db = None
        if using_database:
            db_url = os.path.join(self.base_url, 'temp3/counts.db')
            db = sqlite3.connect(db_url)
            db.execute('''CREATE TABLE IF NOT EXISTS wiki_counts(
                            mention text,
                            entity text
                            )''')

        num_lines = 0
        ment_ent_list = []

        from time import time
        start = time()

        cnt = 0
        crosswiki_path = os.path.join(
            self.base_url, "generic/p_e_m_data/crosswikis_p_e_m.txt"
        )

        with open(crosswiki_path, "r", encoding="utf-8") as f:
            for line in f:
                num_lines += 1
                # if num_lines % 5000000 == 0:
                if num_lines % 5000000 == 0:
                    
                    print("Processed {} lines".format(num_lines))

                    if using_database:
                        c = db.cursor()
                        c.execute("BEGIN TRANSACTION;")
                        c.executemany('''INSERT INTO wiki_counts(mention, entity)
                                                VALUES (?, ?)
                                            ''', ment_ent_list)
                        c.execute("COMMIT;")
                        c.close()

                        print('Writing {} mention/entity combinations in {} seconds'.format(len(ment_ent_list), time()-start)) 
                        start = time()

                        # Reset index to preserve memory.
                        ment_ent_list = [] 

                parts = line.split("\t")
                mention = unquote(parts[0])
                mentions = []
                entities = []
                
                if ("Wikipedia" not in mention) and ("wikipedia" not in mention):
                    
                    if not using_database and mention not in self.wiki_freq:
                        self.wiki_freq[mention] = {}

                    num_ents = len(parts)
                    for i in range(2, num_ents):
                        ent_str = parts[i].split(",")
                        ent_wiki_id = int(ent_str[0])
                        freq_ent = int(ent_str[1])

                        if (
                            ent_wiki_id
                            not in self.wikipedia.wiki_id_name_map["ent_id_to_name"]
                        ):
                            ent_name_re = self.wikipedia.wiki_redirect_id(ent_wiki_id)
                            if (
                                ent_name_re
                                in self.wikipedia.wiki_id_name_map["ent_name_to_id"]
                            ):
                                ent_wiki_id = self.wikipedia.wiki_id_name_map[
                                    "ent_name_to_id"
                                ][ent_name_re]

                        cnt += 1
                        if (
                            ent_wiki_id
                            in self.wikipedia.wiki_id_name_map["ent_id_to_name"]
                        ):
                            if mention not in self.mention_freq:
                                self.mention_freq[mention] = 0
                            self.mention_freq[mention] += freq_ent

                            ent_name = self.wikipedia.wiki_id_name_map[
                                "ent_id_to_name"
                            ][ent_wiki_id].replace(" ", "_")


                            if using_database:
                                mentions.append(mention)
                                entities.append(ent_name)
                                # mentions = [mention] * num_ents
                                # entities = [ int(parts[i].split(",")[0]) for i in range(2, num_ents)]
                                # # frequencies = [ int(parts[i].split(",")[1]) for i in range(2, num_ents)]

                                # temp = [ (m, e) for m,e in zip(mentions, entities)]
                                
                                # ment_ent_list.extend(temp)
                            else:
                                if ent_name not in self.wiki_freq[mention]:
                                    self.wiki_freq[mention][ent_name] = 0
                                self.wiki_freq[mention][ent_name] += freq_ent

                    if using_database:
                        temp = [ (m, e) for m,e in zip(mentions, entities)]
                        ment_ent_list.extend(temp)    

    def __wiki_counts(self, using_database=False):
        """
        Computes mention/entity for a given Wiki dump.

        :return:
        """

        # c = None
        db = None
        if using_database:
            db_url = os.path.join(self.base_url, 'temp3/counts.db')
            db = self.__initialize_wiki_db(db_url)
        # db = sqlite3.connect(db_url, isolation_level=None)

        num_lines = 0
        num_valid_hyperlinks = 0
        disambiguation_ent_errors = 0

        print("Calculating Wikipedia mention/entity occurrences")

        last_processed_id = -1
        exist_id_found = False

        # el['mention'], el['ent_wikiid']

        ment_ent_list = []

        wiki_anchor_files = os.listdir(
            os.path.join(self.base_url, self.wiki_version, "basic_data/anchor_files/")
        )

        from time import time
        start = time()

        for wiki_anchor in wiki_anchor_files:
            wiki_file = os.path.join(
                self.base_url,
                self.wiki_version,
                "basic_data/anchor_files/",
                wiki_anchor,
            )

            with open(wiki_file, "r", encoding="utf-8") as f:
                for line in f:
                    num_lines += 1
                    # if num_lines % 5000000 == 0:
                    if num_lines % 5000000 == 0:

                        # ----------------- NOTE IMPORTANT -----------------------
                        print(
                            "Processed {} lines, valid hyperlinks {}".format(
                                num_lines, num_valid_hyperlinks
                            )
                        )
                        if using_database: 
                            # Execute many.
                            c = db.cursor()
                            c.execute("BEGIN TRANSACTION;")
                            c.executemany('''INSERT INTO wiki_counts(mention, entity)
                                                VALUES (?, ?)
                                            ''', ment_ent_list)
                            c.execute("COMMIT;")
                            c.close()
                            
                            print('Writing {} mention/entity combinations in {} seconds'.format(len(ment_ent_list), time()-start)) 
                            start = time()

                            # Reset index to preserve memory.
                            ment_ent_list = [] 
                        # ----------------- NOTE IMPORTANT -----------------------

                    if '<doc id="' in line:
                        id = int(line[line.find("id") + 4 : line.find("url") - 2])
                        if id <= last_processed_id:
                            exist_id_found = True
                            continue
                        else:
                            exist_id_found = False
                            last_processed_id = id
                    else:
                        if not exist_id_found:
                            (
                                list_hyp,
                                disambiguation_ent_error,
                                print_values,
                            ) = self.__extract_text_and_hyp(line)

                            disambiguation_ent_errors += disambiguation_ent_error
                            if using_database:
                                # Get all mention/entities and extend list.
                                # temp = [ (el['mention'], el['ent_wikiid']) for el in list_hyp ]
                                # ment_ent_list.extend(temp)

                                mentions = []
                                entities = []
                                
                                for el in list_hyp:
                                    mention = el["mention"]
                                    ent_wiki_id = el["ent_wikiid"]

                                    num_valid_hyperlinks += 1

                                    if (
                                        ent_wiki_id
                                        in self.wikipedia.wiki_id_name_map["ent_id_to_name"]
                                    ):
                                        if mention not in self.mention_freq:
                                            self.mention_freq[mention] = 0
                                        self.mention_freq[mention] += 1

                                        ent_name = self.wikipedia.wiki_id_name_map[
                                            "ent_id_to_name"
                                        ][ent_wiki_id].replace(" ", "_")

                                        mentions.append(mention)
                                        entities.append(ent_name)

                                temp = [ (m, e) for m, e in zip(mentions, entities)]
                                ment_ent_list.extend(temp)
                            else:
                                for el in list_hyp:
                                    mention = el["mention"]
                                    ent_wiki_id = el["ent_wikiid"]

                                    num_valid_hyperlinks += 1
                                    if mention not in self.wiki_freq:
                                        self.wiki_freq[mention] = {}

                                    if (
                                        ent_wiki_id
                                        in self.wikipedia.wiki_id_name_map["ent_id_to_name"]
                                    ):
                                        if mention not in self.mention_freq:
                                            self.mention_freq[mention] = 0
                                        self.mention_freq[mention] += 1

                                        ent_name = self.wikipedia.wiki_id_name_map[
                                            "ent_id_to_name"
                                        ][ent_wiki_id].replace(" ", "_")
                                        if ent_name not in self.wiki_freq[mention]:
                                            self.wiki_freq[mention][ent_name] = 0
                                        self.wiki_freq[mention][ent_name] += 1

                                
                            # print(list(self.wiki_freq.items())[:10])


        if using_database:
            db.close()

        print(
            "Done computing Wikipedia counts. Num valid hyperlinks = {}".format(
                num_valid_hyperlinks
            )
        )

    def __extract_text_and_hyp(self, line):
        """
        Extracts hyperlinks from given Wikipedia document to obtain mention/entity counts.

        :return: list of mentions/wiki Ids and their respective counts (plus some statistics).
        """

        line = unquote(line)
        list_hyp = []
        num_mentions = 0
        start_entities = [m.start() for m in re.finditer('<a href="', line)]
        end_entities = [m.start() for m in re.finditer('">', line)]
        end_mentions = [m.start() for m in re.finditer("</a>", line)]

        disambiguation_ent_errors = 0
        start_entity = line.find('<a href="')

        while start_entity >= 0:
            line = line[start_entity + len('<a href="') :]
            end_entity = line.find('">')
            end_mention = line.find("</a>")
            mention = line[end_entity + len('">') : end_mention]

            if (
                ("Wikipedia" not in mention)
                and ("wikipedia" not in mention)
                and (len(mention) >= 1)
            ):
                # Valid mention
                entity = line[0:end_entity]
                find_wikt = entity.find("wikt:")
                entity = entity[len("wikt:") :] if find_wikt == 0 else entity
                entity = self.wikipedia.preprocess_ent_name(entity)

                if entity.find("List of ") != 0:
                    if "#" not in entity:
                        ent_wiki_id = self.wikipedia.ent_wiki_id_from_name(entity)
                        if ent_wiki_id == -1:
                            disambiguation_ent_errors += 1
                        else:
                            num_mentions += 1
                            list_hyp.append(
                                {
                                    "mention": mention,
                                    "ent_wikiid": ent_wiki_id,
                                    "cnt": num_mentions,
                                }
                            )
            # find new entity
            start_entity = line.find('<a href="')
        return (
            list_hyp,
            disambiguation_ent_errors,
            [len(start_entities), len(end_entities), len(end_mentions)],
        )


    def __initialize_custom_db(self, db_url):
        # db_url = os.path.join(self.base_url, 'counts/counts.db')
        #db = sqlite3.connect(db_url, isolation_level=None)
        db = sqlite3.connect(db_url)
        c = db.cursor()

        ## Create custom counts table
        c.execute('''CREATE TABLE IF NOT EXISTS custom_counts(
                        mention text,
                        entity text, 
                        count integer) ''')

        # c.execute('''CREATE TABLE IF NOT EXISTS wiki_counts(
        #                 mention text,
        #                 entity text
        #                 )''')

        # Create index for faster retrieval
        #NOTE: Create index after done inserting on both mention and entity.
        # c.execute('''CREATE INDEX IF NOT EXISTS  custom_mention_index 
        #          ON custom_counts (mention)''')

        # self.counts_db = db
        return db
    
    def __initialize_wiki_db(self, db_url):
        print("Initializing wiki database")

        db = sqlite3.connect(db_url)
        # db = sqlite3.connect(db_url, isolation_level=None)
        c = db.cursor()

        #NOTE: Everytime we compute this again, the table should be cleared (same with custom etc).
        c.execute("BEGIN TRANSACTION;")
        c.execute("DROP TABLE IF EXISTS wiki_counts")
        c.execute("COMMIT;")

        c.execute('''CREATE TABLE IF NOT EXISTS wiki_counts(
                        mention text,
                        entity text
                        )''')

        c.close()
        return db

    def __create_wiki_index(self, db_url):
        print("Creating wiki index")
        db = sqlite3.connect(db_url)
        c = db.cursor()
        c.execute('''CREATE INDEX IF NOT EXISTS  wiki_mention_index 
                     ON wiki_counts (mention)''')
        c.close()
        return db


    def fill_db_from_json(self):
        ## Fill with data from json

        # f = open(os.path.join(self.base_url, 'counts/data.json'))
        # clueweb_dict = json.load(f)
        # lines = 0

        # for mention, entity_dict in clueweb_dict.items():
        #     for entity, count in entity_dict.items():
        #         lines += 1
        #         if (lines % 1000000 == 0):
        #             print("Processed {} lines".format(lines))

        #         c.execute('''INSERT INTO custom_counts(mention, entity, count) 
        #                         VALUES (?, ?, ?)''', (mention, entity, count))
        pass

    def compute_custom_with_db(self, custom=None):
        """
        Computes p(e|m) index for YAGO and combines this index with the Wikipedia p(e|m) index as reported
        by Ganea et al. in 'Deep Joint Entity Disambiguation with Local Neural Attention'.

        Alternatively, users may specificy their own custom p(e|m) by providing mention/entity counts.


        :return:
        """ 
        #TODO enable custom frequencies

        print("Computing p(e|m)")

        db_url = os.path.join(self.base_url, 'counts/counts.db')
        db = self.__initialize_custom_db(db_url)
        c = db.cursor()
        d = db.cursor()
        
        #TODO work in batches
        # e.g. WHERE ROWID BETWEEN 0 AND 1000

        
        c.execute('''
                SELECT mention, COUNT(entity)
                FROM custom_counts
                GROUP BY mention''')
        
        
        for mention, total in c:
            d.execute('''
                    SELECT entity, count
                    FROM custom_counts
                    WHERE mention=?
                        ''', (mention, ))
            data = d.fetchall()

            # Assuming uniform distribution, add to mention_freq & calculate prior
            if mention not in self.mention_freq:
                self.mention_freq[mention] = 0
            self.mention_freq[mention] += 1
            cust_ment_ent_temp = {
                k: 1 / total for k, v in data
            }

            # cust_ment_ent_temp = {}
            # for entity, _ in d:
            #     cust_ment_ent_temp[entity] =  1 / total

            if mention not in self.p_e_m:
                self.p_e_m[mention] = cust_ment_ent_temp
            else:
                for ent_wiki_id in cust_ment_ent_temp:
                    prob = cust_ment_ent_temp[ent_wiki_id]
                    if ent_wiki_id not in self.p_e_m[mention]:
                        self.p_e_m[mention][ent_wiki_id] = 0.0

                    # Assumes addition of p(e|m) as described by authors.
                    self.p_e_m[mention][ent_wiki_id] = np.round(
                        min(1.0, self.p_e_m[mention][ent_wiki_id] + prob), 3
                    )

        db.close()
    
    


    def compute_wiki_with_db(self):
        """
        Computes p(e|m) index for a given wiki and crosswikis dump.

        :return:
        """
        self.__wiki_counts(using_database=True)
        
        self.__cross_wiki_counts(using_database=True)

        # #TODO: move this to the right place
        db_url = os.path.join(self.base_url, 'temp3/counts.db')
        db = self.__create_wiki_index(db_url)
        
        # db = sqlite3.connect(db_url)
    
        c = db.cursor()
        d = db.cursor()

        # Step 1: Calculate p(e|m) for wiki.
        print("Filtering candidates and calculating p(e|m) values for Wikipedia.")

            
        c.execute('''
                SELECT mention
                FROM wiki_counts
                GROUP BY mention''')
        
        # print(c.fetchmany(5))
        num_mentions = 0
        batch_size = 50000

        while True:
            if num_mentions % 500000 == 0:
                print("Processed {} mentions".format(num_mentions))

            batch = c.fetchmany(batch_size)

            if not batch:
                break

            for row in batch:
                ent_mention = row[0]
                if len(ent_mention) < 1:
                    continue
                # print('ent_mention', ent_mention)
                num_mentions += 1

                d.execute('''
                        SELECT entity, COUNT(entity)
                        FROM wiki_counts
                        WHERE mention=?
                        GROUP BY mention, entity
                        ORDER BY COUNT(entity) DESC
                            ''', (ent_mention, ))

                ent_wiki_names = d.fetchall()

                # Get the sum of at most 100 candidates, but less if less are available.
                total_count = np.sum([v for k, v in ent_wiki_names][:100])

                if total_count < 1:
                    continue

                self.p_e_m[ent_mention] = {}

                for ent_name, count in ent_wiki_names:
                    self.p_e_m[ent_mention][ent_name] = count / total_count

                    if len(self.p_e_m[ent_mention]) >= 100:
                        break