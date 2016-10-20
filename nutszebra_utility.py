from __future__ import division, print_function, absolute_import, unicode_literals
import os
import re
import six
import sys
import mmap
import json
from tqdm import tqdm
import threading
import multiprocessing
import subprocess
import numpy as np
import itertools
import six.moves.cPickle as pickle
import collections
from operator import itemgetter
from os.path import expanduser
from six.moves.urllib_parse import urlparse
import nutszebra_basic_bisect


class Command(object):

    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout=3600):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True)
            self.process.communicate()
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.process.terminate()
            thread.join()
        return self.process.returncode


class Histogram(object):

    @staticmethod
    def quantize(img, bit=8):
        img = img.copy()
        border = 256 // bit
        start = six.moves.range(0, 256, border)
        for bit_value, i in enumerate(start):
            floor = i
            ceil = i + border
            indices = np.where((floor <= img) * (img <= ceil))
            img[indices] = bit_value
        return img

    @staticmethod
    def histogram(img, bit=8):
        hist = np.zeros((bit, bit, bit), dtype=np.int).tolist()
        height, width, channel = img.shape
        img = img.reshape((height * width, channel)).tolist()
        for row in img:
            r, g, b = row
            hist[r][g][b] += 1
        return Utility.flat_two_dimensional_list(Utility.flat_two_dimensional_list(hist))


class UnionFind(object):

    """UnifonFind class

    Note:
        http://www.geocities.jp/m_hiroi/light/pyalgo61.html

    Attributes:
        table (list): table for group
    """

    def __init__(self, size):
        # negative value means the number of group
        # positive number is the index of its parent
        self.table = [-1 for _ in six.moves.range(size)]

    # find the root of the group
    def find(self, x):
        if self.table[x] < 0:
            return x
        else:
            # compress the pathway
            self.table[x] = self.find(self.table[x])
            return self.table[x]

    # merge group
    def union(self, x, y):
        s1 = self.find(x)
        s2 = self.find(y)
        if s1 != s2:
            if self.table[s1] <= self.table[s2]:
                # smaller number means that the group has more children
                self.table[s1] += self.table[s2]
                self.table[s2] = s1
            else:
                self.table[s2] += self.table[s1]
                self.table[s1] = s2
            return True
        return False

    # give representive of groups and the number of elements in each group
    def subset(self):
        a = []
        for i in six.moves.range(len(self.table)):
            if self.table[i] < 0:
                a.append((i, -self.table[i]))
        return a

    def relationship(self):
        groups = self.subset()
        answer = [[] for _ in six.moves.range(len(groups))]
        for i, group in enumerate(groups):
            representive, _ = group
            indices = np.where(np.array(self.table) == representive)[0].tolist()
            indices.append(representive)
            answer[i] = indices
        return answer

    def same(self, x, y):
        return self.find(x) == self.find(y)


class Lcs(object):

    @staticmethod
    def lcs(a, b):
        dp = [[0] * (len(b) + 1) for _ in six.moves.range(len(a) + 1)]
        word = [[''] * (len(b) + 1) for _ in six.moves.range(len(a) + 1)]
        element_a = six.moves.range(len(a))
        element_b = six.moves.range(len(b))
        for i, ii in itertools.product(element_a, element_b):
            if a[i] == b[ii]:
                dp[i + 1][ii + 1] = dp[i][ii] + 1
                word[i + 1][ii + 1] = word[i][ii] + a[i]
            else:
                if dp[i][ii + 1] <= dp[i + 1][ii]:
                    dp[i + 1][ii + 1] = dp[i + 1][ii]
                    word[i + 1][ii + 1] = word[i + 1][ii]
                else:
                    dp[i + 1][ii + 1] = dp[i][ii + 1]
                    word[i + 1][ii + 1] = word[i][ii + 1]
        return (dp, word)


class Lis(object):

    @staticmethod
    def lis(array):
        length = len(array)
        dp = [float('inf')] * length
        answer = [[] for _ in six.moves.range(length)]
        for i in six.moves.range(length):
            dp[i] = 1
            answer[i].append(array[i])
            for ii in six.moves.range(0, i):
                if array[ii] <= array[i]:
                    if dp[i] <= dp[ii] + 1:
                        dp[i] = dp[ii] + 1
                        answer[i] = answer[ii] + [array[i]]
        return (dp, answer)

    @staticmethod
    def lis_bisect(array):
        length = len(array)
        dp = [float('inf')] * length
        for i in six.moves.range(length):
            index = nutszebra_basic_bisect.Bisect.find(dp, array[i], side='R')
            dp[index] = array[i]
        return len(dp) - nutszebra_basic_bisect.Bisect.howmany(dp, float('inf'))


class Utility(object):

    """Some utility functions are defined here

    Attributes:
        alphabet_lowercase (list): store all alphabet in lowercase
        alphabet_capital (list): store all alphabet in capital
        numbers (list): store all digits in string
        reg_text (list): regular expression to search for text file
        reg_jpg (list): regular expression to search for jpg
        reg_png (list): regular expression to search for png
        reg_pickle (list): regular expression to search for pickle
        reg_json (list): regular expression to search for json
        home (str): home, e.g. /home/nutszebra
        nutszebra_path (list): path to nutszebra_utility
        inf (inf): inf
    """
    alphabet_lowercase = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    alphabet_capital = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    reg_text = [r'.*\.txt$', r'.*\.text$', r'.*\.TEXT$']
    reg_jpg = [r'.*\.jpg$', r'.*\.jpeg$', r'.*\.JPEG$', r'.*\.Jpeg$']
    reg_png = [r'.*\.png$', r'.*\.Png$', r'.*\.PNG$']
    reg_pickle = [r'.*\.pkl$', r'.*\.pickle$', r'.*\.PKL$', r'.*\.Pickle$']
    reg_json = [r'.*\.json$', r'.*\.Json$', r'.*\.JSON$']
    home = expanduser("~")
    nutszebra_path = home + '/git/nutszebra_research/python/utility/src'
    inf = float('inf')

    def __init__(self):
        pass

    @staticmethod
    def slice_and_paste_prefix(seq, separator='/', start=0, end=None, prefix=''):
        """Slice strings and then put prefix

        Edited date:
            160606

        Example:

        ::
            seq = u'/home/ubuntu/img/shoes/sneakers/m693084193.jpg'
            answer = self.slice_and_paste_prefix(seq, separator='/', start=0, end=None, prefix='')
            >>> print(answer)
                u'/home/ubuntu/img/shoes/sneakers/m693084193.jpg'

            answer = self.slice_and_paste_prefix(seq, separator='/', start=1, end=None, prefix='')
            >>> print(answer)
                u'home/ubuntu/img/shoes/sneakers/m693084193.jpg'

            answer = self.slice_and_paste_prefix(seq, separator='/', start=2, end=None, prefix='')
            >>> print(answer)
                u'ubuntu/img/shoes/sneakers/m693084193.jpg'

            answer = self.slice_and_paste_prefix(seq, separator='/', start=-3, end=None, prefix='/home/suguru/aws_CodeNext/fashion/mercari/')
            >>> print(answer)
                u'/home/suguru/aws_CodeNext/fashion/mercari/shoes/sneakers/m693084193.jpg'

        Args:
            seq (str): strings to slice
            separator (str): separator to slplit seq
            start (str): the index to start slice
            end (Optional[int, None]): the index to end slice
            prefix (str): prefix is added to sliced strings

        Returns:
            str: sliced sequence with prefix
        """

        tmp = seq.split(separator)
        if end is None:
            return prefix + separator.join(tmp[start:end])
        else:
            return prefix + separator.join(tmp[start:end])

    @staticmethod
    def flat_two_dimensional_list(data):
        # http://d.hatena.ne.jp/xef/20121027/p2
        return list(itertools.chain.from_iterable(data))

    @staticmethod
    def pwd():
        return os.path.abspath('./')

    @staticmethod
    def input():
        return six.moves.input()

    @staticmethod
    def input_cast(cast=int):
        return six.moves.map(cast, Utility.input().split())

    def input_multiple_lines(self):
        # http://nemupm.hatenablog.com/entry/2015/01/03/234840
        return sys.stdin.readlines()

    @staticmethod
    def set_recursion_limit(n=1 * 10 ** 8):
        sys.setrecursionlimit(n)
        return True

    @staticmethod
    def get_recursion_limit():
        return sys.getrecursionlimit()

    @staticmethod
    def download_file(url, destination, file_name, timeout=3600, header='--header="Accept: text/html" --user-agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0" '):
        """Download file

        Edited date:
            160429

        Test:
            160429

        Example:

        ::

            self.download_file('http://img5.zozo.jp/goodsimages/502/11989502/11989502B_16_D_500.jpg', '.', 'test.jpg', timeout=10)

        Args:
            url (str): url to download
            destination (str): put the downloaded file onto the destination
            file_name (str): file name
            timeout (Optional[int, float]): If downloading is not finished in timeout seconds, stop downloading
            header (Optional[str]): header for wget
        """

        if destination[-1] is not '/':
            destination = destination + '/'
        cmd = 'wget ' + header + url + ' -O ' + destination + file_name + ' -q'
        command = Command(cmd)
        command.run(timeout=timeout)

    def untar_gz(self, tar_gz_file, destination='./', timeout=3600):
        """Untar gz file

        Edited date:
            160429

        Test:
            160429

        Example:

        ::

            self.untra_gz('./test.gz', '.', timeout=10)

        Args:
            tar_gz_file (str): gz file
            destination (str): untar gz file onto the destination
            file_name (str): file name
            timeout (Optional[int, float]): If untar_gz is not finished in timeout seconds, stop untra_gz
        """

        cmd = 'tar -zxvf ' + tar_gz_file + ' -C ' + destination
        command = Command(cmd)
        command.run(timeout=timeout)

    def tar_gz(self, tar_gz_file, destination='./', timeout=3600):
        """tar gz file

        Edited date:
            160429

        Test:
            160429

        Example:

        ::

            self.untra_gz('./test.gz', '.', timeout=10)

        Args:
            tar_gz_file (str): file to apply tar command
            destination (str): file that tar is applied onto the destination
            file_name (str): file name
            timeout (Optional[int, float]): If untar_gz is not finished in timeout seconds, stop untra_gz
        """

        cmd = 'tar -zcvf ' + tar_gz_file + '.tar.gz ' + tar_gz_file + ' -C ' + destination
        command = Command(cmd)
        command.run(timeout=timeout)

    def url_to_directory(self, url, whole_flag=False):
        """Convert url to directory name

        Edited date:
            160429

        Test:
            160429

        Example:

        ::

            answer = self.url_to_directory('http://img5.zozo.jp/goodsimages/502/11989502/11989502B_16_D_500.jpg')
            >>> print(answer)
                '___a___goodsimages___a___502___a___11989502___a___11989502B_16_D_500___b___jpg'

            answer = self.url_to_directory('http://img5.zozo.jp/goodsimages/502/11989502/11989502B_16_D_500.jpg', whole_flag=True)
            >>> print(answer)
                'http___e______a______a___img5___b___zozo___b___jp___a___goodsimages___a___502___a___11989502___a___11989502B_16_D_500___b___jpg

        Args:
            url (str): url to convert to directory name
            whole_flag (bool): if whole_flag is True, whole url will be converted to the directory name.

        Returns:
            str: converted url
        """
        if whole_flag is True:
            return self._url_to_directory(url)

        scheme, netloc, path, params, query, fragment = urlparse(url)
        # ___a___: /
        # ___b___: .
        # ___c___: ?
        # ___d___: =
        # ___e___: :
        name = ''
        name += self._url_to_directory(path)
        if query:
            name += '___c___' + self._url_to_directory(query)
        return name

    def _url_to_directory(self, sentence):
        """Private method for self.url_to_directory

        Edited date:
            160429

        Args:
            sentence (str): sentence to be converted

        Returns:
            str: converted sentence
        """
        # ___a___: /
        # ___b___: .
        # ___c___: ?
        # ___d___: =
        # ___e___: :
        return sentence.replace('/', '___a___').replace('.', '___b___').replace('?', '___c___').replace('=', '___d___').replace(':', '___e___')

    def directory_to_url(self, directory):
        """Convert directory to url

        Edited date:
            160429

        Test:
            160429

        Example:

        ::

            answer = self.url_to_directory('http://img5.zozo.jp/goodsimages/502/11989502/11989502B_16_D_500.jpg', whole_flag=True)
            >>> print(answer)
                'http___e______a______a___img5___b___zozo___b___jp___a___goodsimages___a___502___a___11989502___a___11989502B_16_D_500___b___jpg

            answer = self.directory_to_url(answer)
            >>> print(answer)
                'http://img5.zozo.jp/goodsimages/502/11989502/11989502B_16_D_500.jpg

        Args:
            directory (str): the directory name that was converted by self.url_to_directory

        Returns:
            str: converted directory
        """
        # ___a___: /
        # ___b___: .
        # ___c___: ?
        # ___d___: =
        name = directory.split('/')[-1]
        return self._directory_to_url(name)

    def _directory_to_url(self, sentence):
        """Private method for self.directory_to_url

        Edited date:
            160429

        Args:
            sentence (str): sentence to be converted

        Returns:
            str: converted sentence
        """

        # ___a___: /
        # ___b___: .
        # ___c___: ?
        # ___d___: =
        # ___e___: :
        return sentence.replace('___e___', ':').replace('___d___', '=').replace('___c___', '?').replace('___b___', '.').replace('___a___', '/')

    def copy_file(self, src, dest, sudo=True):
        if sudo:
            cmd = 'sudo cp ' + src + ' ' + dest
        else:
            cmd = 'cp ' + src + ' ' + dest
        subprocess.call(cmd, shell=True)

    @staticmethod
    def create_progressbar(end, desc='', stride=1, start=0):
        return tqdm(six.moves.range(int(start), int(end), int(stride)), desc=desc, leave=False)

    @staticmethod
    def log_progressbar(sentence):
        tqdm.write(sentence)
        return True

    @staticmethod
    def make_dir_one(path):
        """Make one directory

        Examples:

        ::

            path = './test'
            self.make_dir_one(path)

        Args:
            path (str): path to dir that you'd like to create

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(path):
            os.makedirs(path)
            return True
        return False

    @staticmethod
    def make_dir(path):
        separated_path = path.split('/')
        tmp_path = ''
        for directory in separated_path:
            tmp_path = tmp_path + directory + '/'
            if directory == '.':
                continue
            Utility.make_dir_one(tmp_path)
        return True

    @staticmethod
    def remove_file(path):
        """Remove file

        Examples:

        ::

            path = './test.text'
            self.remove_file(path)

        Args:
            path (str): path to file that you'd like to delete

        Returns:
            bool: True if successful, False otherwise
        """
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def remove_dir(self, path):
        """Remove directory
        Examples:

        ::

            path = './test'
            self.remove_dir(path)

        Args:
            path (str): path to directory that you'd like to delete

        Returns:
            bool: True if successful, False otherwise
        """
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    @staticmethod
    def find_files_recursively(base):
        """Get all files recursively beneath base path

        Edited date:
            160628

        Test:
            160628

        Note:
            directory is disregarded

        Examples:

        ::

            base = './'
            files = self.find_files_recursively(base)
            >>> print(files)
            ['./nutszebra.py',
             './nutszebra_gmail.py',
             './nutszebra_preprocess.py'
             ...
             ]

        Args:
            base (str): base path

        Returns:
            list: return the list that store all files name
        """
        if base[-1] == '/':
            base = base[:-1]
        all_files = []
        walk = os.walk(base, topdown=True, onerror=None, followlinks=False)
        for dirpath, dirs, files in walk:
            for f in files:
                all_files.append(dirpath + '/' + f)
        return all_files

    @staticmethod
    def find_files(path, affix_flag=False):
        """Get all path of files; this function is not recursive

        Edited date:
            160628

        Test:
            160628

        Note:
            directories are included

        Examples:

        ::

            path = './'
            files = self.find_files(path)
            >>> print(files)
            ['./nutszebra.py',
             './nutszebra_gmail.py',
             './nutszebra_preprocess.py'
             ...
             ]

            files = self.find_files(path, affix_flag=True)
            >>> print(files)
            ['nutszebra.py',
             'nutszebra_gmail.py',
             'nutszebra_preprocess.py'
             ...
             ]

        Args:
            path (str): path
            affix_flag (Optional[True, False]): if True, only file or directory names are stored

        Returns:
            list: return list that stores file name paths
        """
        if path[-1] == '/':
            path = path[:-1]
        if affix_flag is False:
            return [path + '/' + name for name in os.listdir(path)]
        else:
            return [name for name in os.listdir(path)]

    @staticmethod
    def reg_extract(files, reg, casting=None):
        """Extract elements by resular expression

        Edited date:
            160628

        Test:
            160628

        Note:
            | reg can be:
            |    nutszebra_utility.Utility.reg_jpg
            |    nutszebra_utility.Utility.reg_png
            |    nutszebra_utility.Utility.reg_text
            |    nutszebra_utility.Utility.reg_json
            |    nutszebra_utility.Utility.reg_pickle

        Examples:

        ::

            data = ['1.jpg', '3.png', '.4.swp', 'hi', '10.jpeg', '2.Jpeg', '4.JPEG', '5.PNG', '6.Png', 'gomi', 'hoge', '1.txt', '2.TEXT', '3.text', '5.Json', '6.JSON', '7.json', 'hyohyo', '10.Pickle', 'a', '1.pickle', '11.Pickle']
            >>> print(self.reg_extract(data, self.reg_jpg))
                ['1.jpg', '10.jpeg', '4.JPEG', '2.Jpeg']
            >>> print(self.reg_extract(data, self.reg_png))
                ['3.png', '6.Png', '5.PNG']
            >>> print(self.reg_extract(data, self.reg_text))
                ['1.txt', '3.text', '2.TEXT']
            >>> print(self.reg_extract(data, self.reg_json))
                ['7.json', '5.Json', '6.JSON']
            >>> print(self.reg_extract(data, self.reg_pickle))
                ['1.pickle', '10.Pickle', '11.Pickle']

            data = ['value', '0.001', 'value1', '3.23', '4', 'end']
            reg = [r'^\d+\.\d+$|^\d+$']
            >>> print(self.reg_extract(data, reg))
                ['0.001', '3.23', '4']
            >>> print(self.reg_extract(data, reg, self.cast_int))
                [0, 3, 4]
            >>> print(self.reg_extract(data, reg, self.cast_float))
                [0.001, 3.23, 4.0]

        Args:
            files (list): files to extract
            reg (list): regular expression
            casting Optional([nutszebra_utility.Utility.cast_int, nutszebra_utility.Utility.cast_float, nutszebra_utility.Utility.cast_str, None]): extracted elements can be casted

        Returns:
            list: extracted elements
        """
        if casting is None:
            return [ff for r in reg for f in files for ff in re.findall(r, f)]
        else:
            return [casting(ff)[0] for r in reg for f in files for ff in re.findall(r, f)]

    @staticmethod
    def cast_int(array):
        """Convert values inside array to int

        Edited date:
            160628

        Test:
            160628

        Examples:

        ::

            data = [1.0, 2.0, 3.0]
            >>> print(self.cast_int(data))
                [1, 2, 3]

        Args:
            array (list): array that contains value to be converted

        Returns:
            list: converted array
        """
        if type(array) is str or not isinstance(array, collections.Iterable):
            array = [array]
        # int('1.0') gives error, thus float first
        return [int(float(num)) for num in array]

    @staticmethod
    def cast_float(array):
        """Convert values inside array to float

        Edited date:
            160628

        Test:
            160628

        Examples:

        ::

            data = [1, 2, 3]
            >>> print(self.cast_float(data))
                [1.0, 2.0, 3.0]

        Args:
            array (list): array that contains value to be converted

        Returns:
            list: converted array
        """
        if type(array) is str or not isinstance(array, collections.Iterable):
            array = [array]
        return [float(num) for num in array]

    @staticmethod
    def cast_str(array):
        """Convert values inside array to str

        Edited date:
            160628

        Test:
            160628

        Examples:

        ::

            data = [1, 2, 3]
            >>> print(self.cast_str(data))
                ['1', '2', '3']

        Args:
            array (list): array that contains value to be converted

        Returns:
            list: converted array
        """
        if type(array) is str or not isinstance(array, collections.Iterable):
            array = [array]
        return [str(num) for num in array]

    @staticmethod
    def save_pickle(data, path):
        """Save as pickle

        Edited date:
            160628

        Test:
            160628

        Examples:

        ::

            data = [numpy.zeros((10,10)),numpy.zeros((10,10))]
            path = 'test.pickle'
            self.save_pickle(data, path)

        Args:
            data : data to save
            path (str): path

        Returns:
            bool: True if successful, False otherwise
        """
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return True

    @staticmethod
    def load_pickle(path):
        """Load pickle

        Edited date:
            160628

        Test:
            160628

        Examples:

        ::

            path = 'test.pickle'
            data = self.load_pickle(path)

        Args:
            path (str): pickle file

        Returns:
            loaded pickle
        """
        with open(path, 'rb') as f:
            answer = pickle.load(f)
        return answer

    @staticmethod
    def save_json(data, path, jap=False):
        """Save as json

        Edited date:
            160628

        Test:
            160628

        Examples:

        ::

            data = {'a':1, 'b':2}
            path = 'test.json'
            self.save_json(data, path)

        Args:
            data (dict): data to save
            path (str): path

        Returns:
            bool: True if successful, False otherwise
        """

        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=not jap)
        return True

    @staticmethod
    def load_json(path):
        """Load json

        Edited date:
            160628

        Test:
            160628

        Examples:

        ::

            path = 'test.json'
            data = self.load_json(path)

        Args:
            path (str): json file

        Returns:
            dict: loaded json
        """

        with open(path, 'r') as f:
            answer = json.load(f)
        return answer

    @staticmethod
    def count_line(path):
        """count lines of text

        Edited date:
            160626

        Test:
            160626

        Examples:

        ::

            path = 'test.text'
            self.count_line(text)

        Args:
            path (str): text file

        Returns:
            int: number of lines
        """
        f = os.open(path, os.O_RDONLY)
        buf = mmap.mmap(f, 0, prot=mmap.PROT_READ)
        answer = 0
        readline = buf.readline
        while readline():
            answer += 1
        buf.close()
        return int(answer)

    @staticmethod
    def save_text(data, output):
        """Save as text

        Edited date:
            160626

        Test:
            160626

        Examples:

        ::

            data = ['this', 'is', 'test']
            output = 'test.text'
            self.save_text(data, output)

        Args:
            data (list): data to save
            output (str): output name

        Returns:
            bool: True if successful, False otherwise
        """
        if not type(data) == list:
            data = [data]
        with open(output, 'w') as f:
            for i in six.moves.range(len(data)):
                f.write(data[i] + '\n')
        return True

    @staticmethod
    def load_text(path, count=False):
        """Load text

        Edited date:
            160626

        Test:
            160626

        Note:
            | load big file
            | without count: 394.2885489463806
            | count: 529.5308997631073

        Examples:

        ::

            path = 'test.text'
            count = True
            data = self.load_text(path)

        Args:
            path (str): json file
            count (bool): if True, firstly count lines of files and initialized list

        Returns:
            list: content inside text
        """
        if count is True:
            num = Utility.count_line(path)
            content = [0] * num
            progressbar = Utility.create_progressbar(num, 'loading text...')
            for i, line in six.moves.zip(progressbar, Utility.yield_text(path)):
                content[i] = line
            return content
        else:
            content = []
            for line in Utility.yield_text(path):
                content.append(line)
            return content

    @staticmethod
    def yield_text(path, cast=str):
        """Yield each line of text

        Edited date:
            160626

        Test:
            160626

        Note:
            \n at the end of lines are removed

        Args:
            path (str): path to text file
            cast Optional([int, float, str,...]): casting

        Yields:
            str: one line of text
        """
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                # remove \n
                yield cast(line[:-1])
                line = f.readline()

    @staticmethod
    def _itemgetter(order):
        """Give itemgetter for sort

        Edited date:
            160626

        Test:
            160626

        Args:
            order (tuple): sort order is stored

        Returns:
            function: itemgetter
        """
        if type(order) is float:
            order = int(order)
        if type(order) is int:
            order = [order]
        return itemgetter(*order)

    @staticmethod
    def sort_list(array, key=(0), reverse=False):
        """Sort tuples in array

        Edited date:
            160626

        Test:
            160626

        Examples:

        ::

            data = [('a', 1), ('b', 5), ('c', 2)]
            sorted_list = self.sort_list(data, key=(1))
            >>> print(sorted_list)
            [('a', 1), ('c', 2), ('b', 5)]

            data = [('02', 2), ('02', 1), ('01', 3)]
            sorted_list = self.sort_list(data, key=(0, 1))
            >>> print(sorted_list)
            [('01', 3), ('02', 1), ('02', 2)]

        Args:
            array (list): data to sort
            key (int): sort by array[i][key]
            reverse (bool): reverse flag

        Returns:
            list: sorted list
        """
        return sorted(array, key=Utility._itemgetter(key), reverse=reverse)

    @staticmethod
    def get_cpu_number():
        """Get cpu number

        Edited date:
            160620

        Test:
            160626

        Example:

        ::

            test = self.get_cpu_number()
            >>> print(test)
            2
            int: cpu number
        """

        return int(multiprocessing.cpu_count())
