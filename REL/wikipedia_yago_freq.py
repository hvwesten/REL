import os
import sqlite3
import json
import re
from urllib.parse import unquote

import numpy as np

from REL.db.generic import GenericLookup
from REL.utils import first_letter_to_uppercase, trim1, unicode2ascii

from REL.mapping.freebasewikimapper import FreebaseWikiMapper
import tarfile

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

        self.wiki_db_url = ""
        self.custom_db_url = ""
        self.mapper = FreebaseWikiMapper(self.base_url,'./mapping/index_enwiki-20190420.db', './mapping/index_fb2wp.db')

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

        # self.__wiki_counts()
        self.__cross_wiki_counts()

        fpath = os.path.join(self.base_url, 'temp/zone2/cw.json')
        with open(fpath, 'w') as f:
            json.dump(self.wiki_freq, f)


        # Step 1: Calculate p(e|m) for wiki.
        # print("Filtering candidates and calculating p(e|m) values for Wikipedia.")
        # for ent_mention in self.wiki_freq:
        #     if len(ent_mention) < 1:
        #         continue

        #     ent_wiki_names = sorted(
        #         self.wiki_freq[ent_mention].items(), key=lambda kv: kv[1], reverse=True
        #     )
        #     # Get the sum of at most 100 candidates, but less if less are available.
        #     total_count = np.sum([v for k, v in ent_wiki_names][:100])

        #     if total_count < 1:
        #         continue

        #     self.p_e_m[ent_mention] = {}

        #     for ent_name, count in ent_wiki_names:
        #         self.p_e_m[ent_mention][ent_name] = count / total_count

        #         if len(self.p_e_m[ent_mention]) >= 100:
        #             break
        
        # fpath = os.path.join(self.base_url, 'temp/wf/wiki_freq.json')
        # with open(fpath, 'w') as f:
        #     json.dump(self.wiki_freq, f)


        # del self.wiki_freq

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
                    code = ent_name[x: x + 6]
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
            db = sqlite3.connect(self.wiki_db_url)

        num_lines = 0
        ment_ent_list = []

        from time import time
        start = time()

        cnt = 0
        crosswiki_path = os.path.join(
            self.base_url, "generic/p_e_m_data/crosswikis_p_e_m.txt"
        )

        it = 0
        with open(crosswiki_path, "r", encoding="utf-8") as f:
            for line in f:
                num_lines += 1

                if num_lines % 5000000 == 0:
                    print("Processed {} lines".format(num_lines))
                
                    if using_database:
                        c = db.cursor()
                        c.execute("BEGIN TRANSACTION;")
                        c.executemany('''INSERT INTO wiki_counts(mention, entity, count)
                                                    VALUES (?, ?, ?)
                                                    ON CONFLICT(mention, entity)
                                                    DO UPDATE SET count=count + ?
                                                ''', ment_ent_list)
                        c.execute("COMMIT;")
                        c.close()

                        
                        print('Writing {} mention/entity combinations in {} seconds'.format(
                            len(ment_ent_list), time()-start))
                        start = time()

                        # Reset index to preserve memory.
                        ment_ent_list = []


                parts = line.split("\t")
                mention = unquote(parts[0])

                mentions = []
                entities = []
                counts = []


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
                            ent_name_re = self.wikipedia.wiki_redirect_id(
                                ent_wiki_id)
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
                                counts.append(freq_ent)
                            else:
                                if ent_name not in self.wiki_freq[mention]:
                                    self.wiki_freq[mention][ent_name] = 0
                                self.wiki_freq[mention][ent_name] += freq_ent

                if using_database:
                    #TODO clean up this code
                    temp = [(m, e, c1, c2) for m, e, c1, c2 in zip(mentions, entities, counts, counts)]
                    ment_ent_list.extend(temp)

        if using_database:
            # Add the last few mentions and close the database
            c = db.cursor()
            c.execute("BEGIN TRANSACTION;")
            c.executemany('''INSERT INTO wiki_counts(mention, entity, count)
                                        VALUES (?, ?, ?)
                                        ON CONFLICT(mention, entity)
                                        DO UPDATE SET count=count + ?
                                    ''', ment_ent_list)
            c.execute("COMMIT;")
            c.close()
            db.close()

    def __wiki_counts(self, using_database=False):
        """
        Computes mention/entity for a given Wiki dump.

        :return:
        """

        db = None
        if using_database:
            db = sqlite3.connect(self.wiki_db_url)
            print(f"Using database at url: {self.wiki_db_url}")

        num_lines = 0
        num_valid_hyperlinks = 0
        disambiguation_ent_errors = 0

        print("Calculating Wikipedia mention/entity occurrences")

        last_processed_id = -1
        exist_id_found = False

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
                            c.executemany('''INSERT INTO wiki_counts(mention, entity, count)
                                                    VALUES (?, ?, 1)
                                                    ON CONFLICT(mention, entity)
                                                    DO UPDATE SET count=count+1
                                                ''', ment_ent_list)
                            c.execute("COMMIT;")
                            print('Writing {} mention/entity combinations in {} seconds'.format(
                                len(ment_ent_list), time()-start))
                            start = time()

                            # Reset index to preserve memory.
                            ment_ent_list = []
                        # ----------------- NOTE IMPORTANT -----------------------

                    if '<doc id="' in line:
                        id = int(line[line.find("id") + 4: line.find("url") - 2])
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

                                temp = [(m, e) for m, e in zip(mentions, entities)]
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
            line = line[start_entity + len('<a href="'):]
            end_entity = line.find('">')
            end_mention = line.find("</a>")
            mention = line[end_entity + len('">'): end_mention]

            if (
                ("Wikipedia" not in mention)
                and ("wikipedia" not in mention)
                and (len(mention) >= 1)
            ):
                # Valid mention
                entity = line[0:end_entity]
                find_wikt = entity.find("wikt:")
                entity = entity[len("wikt:"):] if find_wikt == 0 else entity
                entity = self.wikipedia.preprocess_ent_name(entity)

                if entity.find("List of ") != 0:
                    if "#" not in entity:
                        ent_wiki_id = self.wikipedia.ent_wiki_id_from_name(
                            entity)
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

    def __clueweb_counts(self):
        ''' Gets ClueWeb counts as dictionary, assumes that you 
            have a directory called 'ClueWeb09' within the 
            original file structure

            .
            |-- generic
            |-- wiki2019
            |-- ClueWeb09    <---

        '''
        
        print("Getting ClueWeb counts")

        lines_read = 0
        
        clueweb_dict = {}
        clueweb_url = os.path.join(
            self.base_url, 'ClueWeb09/')

        for _, _, files in os.walk(clueweb_url):
            for fn in files:
                if fn.endswith('.tgz'):
                    file_path = os.path.join(clueweb_url, fn)

                    # print(f"Filepath: {file_path}")
                    with tarfile.open(file_path, 'r:gz') as tf:
                        for entry in tf:
                            if entry.name.endswith('.tsv'):
                                f = tf.extractfile(entry)
                                for l in f:
                                    line = str(l, 'utf-8')

                                    lines_read += 1
                                    
                                    # if lines_read % 500000 == 0:
                                    if lines_read % 5000000 == 0:
                                        print(
                                            "Processed {} lines".format(lines_read))
                                        # return clueweb_dict

                                    line = line.rstrip()
                                    line = unquote(line)
                                    parts = line.split('\t')
                                    mention = parts[2]
                                    entity_mid = parts[7]
                                        
                                    wp_titles = self.mapper.id_to_titles(entity_mid)
                                    ent_name = ''
                                    if len(wp_titles) == 0:
                                        ent_name = '<>' # 
                                    else:    
                                        ent_name = self.mapper.id_to_titles(entity_mid)[0] # get first result

                                    ent_name = self.wikipedia.preprocess_ent_name(ent_name)

                                    if ent_name in self.wikipedia.wiki_id_name_map["ent_name_to_id"]:
                                        if mention not in clueweb_dict:
                                            clueweb_dict[mention] = {}
                                            
                                        ent_name = ent_name.replace(" ", "_")

                                        if ent_name not in clueweb_dict[mention]:
                                            clueweb_dict[mention][ent_name] = 1
                                        else:
                                            clueweb_dict[mention][ent_name] += 1

        with open('./data/cc.json', 'w') as outfile:
            json.dump(clueweb_dict, outfile)  

        return clueweb_dict

    def __initialize_custom_db(self, db_url):
        db = sqlite3.connect(db_url)
        c = db.cursor()

        # Reset table for each run
        c.execute("BEGIN TRANSACTION;")
        c.execute("DROP TABLE IF EXISTS custom_counts")
        c.execute("COMMIT;")

        # Create custom counts table
        c.execute('''CREATE TABLE IF NOT EXISTS custom_counts(
                        mention text,
                        entity text, 
                        count integer) ''')

        print("Initializing custom database...")

        # Insert data into table (TODO make this more efficient)
        clueweb_dict = self.__clueweb_counts()


        # with open(os.path.join(self.base_url, 'data_facc1_english.json')) as clueweb_json:
            # clueweb_dict = json.load(clueweb_json)

        n = 0
        for mention, entity_dict in clueweb_dict.items():
            n+=1
            if n % 500000 == 0:
                print("Processed {} mentions so far.".format(n))
                    
            # for entity, count in entity_dict.items():

                # c.execute("BEGIN TRANSACTION;")
                # c.execute('''INSERT INTO custom_counts(mention, entity, count)
                #             VALUES (?, ?, ?)''', (mention, entity, count))
                # c.execute("COMMIT;")

            mention_list = [mention] * len(entity_dict)
            ent_list = [ e for e, _ in entity_dict.items()]
            cnt_list = [ c for _, c in entity_dict.items()]
            # ment_ent_list = [(m, e, c) for m, e, c in zip(mention_list, ent_list, cnt_list)]
            # print(ment_ent_list)
            c.execute("BEGIN TRANSACTION;")
            c.executemany('''INSERT INTO custom_counts(mention, entity, count)
                            VALUES (?, ?, ?)''', zip(mention_list, ent_list, cnt_list))
            c.execute("COMMIT;")

        # Create index for faster retrieval
        # NOTE: Create index after done inserting on both mention and entity.
        c.execute('''CREATE INDEX IF NOT EXISTS  custom_mention_index 
                 ON custom_counts (mention, entity)''')
        c.close()

        db.close()

    def __create_wiki_db(self, db_url):
        print("Creating wiki database")

        db = sqlite3.connect(db_url)
        c = db.cursor()

        # NOTE: Everytime we compute this again, the table should be cleared (same with custom etc).
        c.execute("BEGIN TRANSACTION;")
        c.execute("DROP TABLE IF EXISTS wiki_counts")
        c.execute("COMMIT;")
        
        c.execute('''CREATE TABLE IF NOT EXISTS wiki_counts(
                        mention text,
                        entity text, 
                        count integer,
                        UNIQUE(mention, entity)) ''')

        c.close()
        db.close()
        # return db

    def __create_wiki_index(self, db_url):
        print("Creating wiki index")
        db = sqlite3.connect(db_url)
        c = db.cursor()
        c.execute('''CREATE INDEX IF NOT EXISTS  wiki_mention_index 
                     ON wiki_counts (mention)''')
        c.close()
        return db

    def __save_pem_to_file(self):
        fpath = os.path.join(self.base_url, 'pem_custom_nodb.json')
        with open(fpath, 'w') as f:
            json.dump(self.p_e_m, f)

    def compute_custom_with_db(self, custom=None):
        """
        Computes p(e|m) index for YAGO and combines this index with the Wikipedia p(e|m) index as reported
        by Ganea et al. in 'Deep Joint Entity Disambiguation with Local Neural Attention'.

        Alternatively, users may specificy their own custom p(e|m) by providing mention/entity counts.


        :return:
        """
        # TODO enable custom frequencies

        print("Computing p(e|m)")

        self.custom_db_url =  os.path.join(self.base_url, "counts/counts.db")

        db_url = os.path.join(self.base_url, self.custom_db_url)
        
        self.__initialize_custom_db(db_url)

        # dc = self.__clueweb_counts()
        
        # db = sqlite3.connect(db_url)
        # db.close()
        # c = db.cursor()
        # d = db.cursor()
        # # print(c.fetchmany(10))
        # c.execute('''
        #         SELECT mention, COUNT(entity)
        #         FROM custom_counts
        #         GROUP BY mention''')

        # num_mentions = 0
        # batch_size = 50000

        # while True:
        #     num_mentions += 1
        #     if num_mentions % 500000 == 0:
        #         print("Processed {} custom mentions".format(num_mentions))

        #     batch = c.fetchmany(batch_size)

        #     if not batch:
        #         break

        #     for mention, total in batch:
        #         d.execute('''
        #                 SELECT entity, count
        #                 FROM custom_counts
        #                 WHERE mention=?
        #                     ''', (mention, ))

        #         data = d.fetchall()

        #         # Assuming uniform distribution, add to mention_freq & calculate prior
        #         if mention not in self.mention_freq:
        #             self.mention_freq[mention] = 0
        #         self.mention_freq[mention] += 1

        #         cust_ment_ent_temp = {
        #             k: 1 / total for k, v in data
        #         }

        #         # cust_ment_ent_temp = {}
        #         # for entity, _ in d:
        #         #     cust_ment_ent_temp[entity] =  1 / total

        #         if mention not in self.p_e_m:
        #             self.p_e_m[mention] = cust_ment_ent_temp
        #         else:
        #             for ent_wiki_id in cust_ment_ent_temp:
        #                 prob = cust_ment_ent_temp[ent_wiki_id]
        #                 if ent_wiki_id not in self.p_e_m[mention]:
        #                     self.p_e_m[mention][ent_wiki_id] = 0.0

        #                 # Assumes addition of p(e|m) as described by authors.
        #                 self.p_e_m[mention][ent_wiki_id] = np.round(
        #                     min(1.0, self.p_e_m[mention][ent_wiki_id] + prob), 3
        #                 )

        # db.close()

    def compute_wiki_with_db(self):
        """
        Computes p(e|m) index for a given wiki and crosswikis dump using sqlite3

        :return:
        """

        # Get path to the database that will contain the Wiki mention/entity pairs
        self.wiki_db_url =  os.path.join(self.base_url, "temp/wiki.db") #wiki_db)    

        # Create the wiki table, emptying it if it already exists
        self.__create_wiki_db(self.wiki_db_url)

        # Compute mention/entity for a given Wiki dump and add it to the database
        self.__wiki_counts(using_database=True)
        
        # Update mention/entity for Wiki with CrossWiki corpus, updating the database
        self.__cross_wiki_counts(using_database=True)
        
        # Creating an index for the Wiki table for faster retrieval
        self.__create_wiki_index(self.wiki_db_url)
        
        # (Re-) Connect to the database
        db = sqlite3.connect(self.wiki_db_url)

        c = db.cursor()
        d = db.cursor()

        # Step 1: Calculate p(e|m) for wiki.
        print("Filtering candidates and calculating p(e|m) values for Wikipedia.")


        # Get all the unique mentions in the table
        c.execute('''
                  SELECT DISTINCT mention
                  FROM wiki_counts''')

        # Initialize 
        num_mentions = 0
        batch_size = 50000

        while True:
            # Get a batch of the mentions
            batch = c.fetchmany(batch_size)
            
            # If the batch is empty, we are done
            if not batch:
                break

            # For each row (= mention )
            for row in batch:
                num_mentions += 1
                if num_mentions % 500000 == 0:
                    print("Processed {} wiki mentions".format(num_mentions))

                # Get the mention, and stop if it is an empty mention
                ent_mention = row[0]
                if len(ent_mention) < 1:
                    continue
                    
                # Get all the entity + count combinations for the mention
                d.execute('''
                          SELECT entity, count
                          FROM wiki_counts
                          WHERE mention=?
                          ORDER BY count DESC
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
