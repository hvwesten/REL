import os
import sqlite3
import json
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
            os.path.join(self.base_url, self.wiki_version, "generated_01"),
            table_name="wiki",
            columns={"p_e_m": "blob", "lower": "text", "freq": "INTEGER"},
        )

        wiki_db.load_wiki(self.p_e_m, self.mention_freq, batch_size=50000, reset=True)

    def compute_wiki(self, add_custom=None):
        """
        Computes p(e|m) index for a given wiki and crosswikis dump.
        :return:
        """

        self.__wiki_counts()

        # Combine with clueweb or YAGO
        if add_custom:
            self.__load_custom(add_custom)
        else:
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
                            # TODO clean up this code
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

    def __load_custom_db(self, custom=None):
        ''' Loads the custom counts in the given json file
            into a database
        '''
             
        print("Loading custom counts from json file")
        custom_list = []

        db = sqlite3.connect(self.custom_db_url)
        c = db.cursor()

        n = 0
        for mention, entity_dict in custom.items():
            n += 1
            if n % 500000 == 0:
                print("Processed {} mentions so far.".format(n))
                
                c.execute("BEGIN TRANSACTION;")
                c.executemany('''INSERT INTO custom_counts(mention, entity, count)
                                            VALUES (?, ?, ?)
                                            ON CONFLICT(mention, entity)
                                            DO UPDATE SET count=count + ?
                                        ''', custom_list)
                c.execute("COMMIT;")

                custom_list = []
        
            mentions = []
            entities = []
            counts = []

            for entity, count in entity_dict.items():
                # Process wikipedia title 
                ent_name = self.wikipedia.preprocess_ent_name(entity)

                # Only add the mention if it is in the KB
                if ent_name in self.wikipedia.wiki_id_name_map["ent_name_to_id"]:

                    ent_name = ent_name.replace(" ", "_")

                    mentions.append(mention)
                    entities.append(ent_name)
                    counts.append(count)
                

            temp = [(m, e, c1, c2) for m, e, c1, c2 in zip(mentions, entities, counts, counts)]   
            custom_list.extend(temp)
        
        # Insert last few mentions into database
        c.execute("BEGIN TRANSACTION;")
        c.executemany('''INSERT INTO custom_counts(mention, entity, count)
                                    VALUES (?, ?, ?)
                                    ON CONFLICT(mention, entity)
                                    DO UPDATE SET count=count + ?
                                ''', custom_list)
        c.execute("COMMIT;")
        
        c.close()
        db.close()

    def __create_custom_db(self):
        print("Creating custom database")

        db = sqlite3.connect(self.custom_db_url)
        c = db.cursor()

        # Reset table for each run
        c.execute("BEGIN TRANSACTION;")
        c.execute("DROP TABLE IF EXISTS custom_counts")
        c.execute("COMMIT;")

        # Create custom counts table
        c.execute('''CREATE TABLE IF NOT EXISTS custom_counts(
                        mention text,
                        entity text, 
                        count integer,
                        UNIQUE(mention, entity)) ''')

        c.close()
        db.close()

    def __create_wiki_db(self):
        print("Creating wiki database")

        db = sqlite3.connect(self.wiki_db_url)
        c = db.cursor()

        # Reset table for each run
        c.execute("BEGIN TRANSACTION;")
        c.execute("DROP TABLE IF EXISTS wiki_counts")
        c.execute("COMMIT;")
        
        # Create wiki table
        c.execute('''CREATE TABLE IF NOT EXISTS wiki_counts(
                        mention text,
                        entity text, 
                        count integer,
                        UNIQUE(mention, entity)) ''')

        c.close()
        db.close()

    def __create_custom_index(self):
        print("Creating custom index")
        db = sqlite3.connect(self.custom_db_url)
        c = db.cursor()

        # Create index for faster retrieval
        c.execute('''CREATE INDEX IF NOT EXISTS  custom_mention_index 
                 ON custom_counts (mention, entity)''')

        c.close()
        db.close()

    def __create_wiki_index(self):
        print("Creating wiki index")

        db = sqlite3.connect(self.wiki_db_url)
        c = db.cursor()
        c.execute('''CREATE INDEX IF NOT EXISTS  wiki_mention_index 
                     ON wiki_counts (mention)''')
        c.close()
        db.close()

    def compute_custom_with_db(self, custom=None):
        """
        Computes p(e|m) index for YAGO and combines this index with the Wikipedia p(e|m) index as reported
        by Ganea et al. in 'Deep Joint Entity Disambiguation with Local Neural Attention'.

        Alternatively, users may specificy their own custom p(e|m) by providing mention/entity counts.


        :return:
        """
        # Get path to the database that will contain the custom mention/entity pairs
        self.custom_db_url =  os.path.join(self.base_url, "counts/custom.db")

        # Create the custom table, emptying it if it already exists
        self.__create_custom_db()
        
        # Load database from file, or throw NOT IMPLEMENTED error
        if custom:
            self.__load_custom_db(custom)
        else:
            print("NOT IMPLEMENTED. You should provide the custom counts as a .json file.")
            print("For ClueWeb these can be calculated using `00_clueweb_to_json`")

            exit(0)

        # Create an index for the custom table for faster retrieval
        self.__create_custom_index()
        
        # (Re-) Connect to the database
        db = sqlite3.connect(self.custom_db_url)
        c = db.cursor()
        d = db.cursor()

        # Retrieve all mentions and their total number of entity counts
        c.execute('''
                SELECT mention, COUNT(entity)
                FROM custom_counts
                GROUP BY mention''')

        num_mentions = 0
        batch_size = 5000

        print("Computing p(e|m)")

        while True:
            # Retrieve a batch of mention/entity total count combinations
            batch = c.fetchmany(batch_size)
            
            # If the batch is empty, we are done
            if not batch:
                break
            
            # For each mention and its total entity count
            for mention, total in batch:
                num_mentions += 1
                if num_mentions % 500000 == 0:
                    print("Processed {} custom mentions".format(num_mentions))

                # Get the actual entities and their counts for the mention
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
        
        c.close()
        d.close()
        db.close()

    def compute_wiki_with_db(self):
        """
        Computes p(e|m) index for a given wiki and crosswikis dump using sqlite3

        :return:
        """

        # Get path to the database that will contain the Wiki mention/entity pairs
        self.wiki_db_url =  os.path.join(self.base_url, "counts/wiki.db") 

        # Create the wiki table, emptying it if it already exists
        self.__create_wiki_db()

        # Compute mention/entity for a given Wiki dump and add it to the database
        self.__wiki_counts(using_database=True)
        
        # Update mention/entity for Wiki with CrossWiki corpus, updating the database
        self.__cross_wiki_counts(using_database=True)
        
        # Create an index for the Wiki table for faster retrieval
        self.__create_wiki_index()
        
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