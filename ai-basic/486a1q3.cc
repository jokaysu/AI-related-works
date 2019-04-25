#include <iostream>
#include <vector>
#include <utility>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <algorithm>

#define maxVer 100
#define maxTrial 100
#define maxNum 100000000

using namespace std;

int numVer;
vector< pair< vector<int>, double > > data[maxVer];
int globalOrder[maxVer + 1];
double globalCost = maxNum;

double costUpdate(int a[], int n) {
	double totalCost = 0;
	for (int i = 1; i <= n; ++i) {
		double minCost = maxNum;
		for (vector< pair<vector<int>, double> >::iterator it = data[a[i]].begin();
		 it != data[a[i]].end(); ++it) {
			bool flag = true;
			for (vector<int>::iterator it2 = it->first.begin(); it2 != it->first.end(); ++it2) {
				bool flag2 = false;
				for (int j = 0; j < i; ++j) {
					if (a[j] == *it2) {
						flag2 = true;
						break;
					}
				}//j
				if (flag2 == false) {
					flag = false;
					break;
				}
			}//int vec
			if (flag) {
				if (it->second < minCost) 
					minCost = it->second;
			}
		}//pair vec
		totalCost += minCost;
	}//i
	if (totalCost < globalCost) {
		globalCost = totalCost;
		for (int i = 0; i <= n; ++i) 
			globalOrder[i] = a[i];
	}
	return totalCost;
}

int main(int argc, char * argv []) {

	//check argc
	if (argc != 2) {
		cerr << "Usage: cs486a1q3 inputfile" << endl;
		return 1;
	}

	//freopen
	freopen(argv[1], "r", stdin);

	//read
	cin >> numVer;
	int readInt, readLine;
	char readChar;
	double readDouble;
	for (int ii = 1; ii <= numVer; ++ii) {
		cin >> readInt;
		if (readInt != ii) {
			cerr << "Reading Error 1" << endl;
			return 1;
		}
		cin >> readLine;
		for (int i = 0; i < readLine; ++i) {
			cin >> readChar;
			if (readChar != '{') {
				cerr << "Reading Error 2 " << readChar << endl;
				return 1;
			}
			vector<int> tempVector;
			double tempDouble;
			data[ii].push_back(make_pair(tempVector, tempDouble));
			while (true) {
				cin >> readInt;
				if (cin.fail()) {
					cin.clear();
					cin >> readChar;
					if (readChar == '}') {
						data[ii][i].first.push_back(0); // empty
						break;
					} else {
						cerr << "Reading Error 3 " << readChar 
							<< " " << ii << " " << i << endl;
						return 1;
					}
				}
				data[ii][i].first.push_back(readInt);
				cin >> readChar;
				if (readChar == '}') break;
				else if (readChar != ',') {
					cerr << "Reading Error 4" << endl;
					return 1;
				}
			}//while
			cin >> readDouble;
			data[ii][i].second = readDouble;
		}//for readLine
	}//for vertices

	//trial loop
	int s[numVer + 1], r[numVer + 1];
	for (int i = 0; i <= numVer; ++i) 
		s[i] = i;
	double cs, cr;
	for (int trial = 0; trial < maxTrial; ++trial) {

		//all random init
		srand(time(NULL));
		random_shuffle(s + 1, s + 1 + numVer);
		cs = costUpdate(s, numVer);

		int k = 0;
		do {

			//swap adj neighbour
			int temp[numVer + 1];
			cr = maxNum;
			for (int i = 1; i < numVer; ++i) {
				//swap adj
				for (int j = 0; j < i; ++j)
					temp[j] = s[j];
				temp[i] = s[i + 1];
				temp[i + 1] = s[i];
				for (int j = i + 2; j <= numVer; ++j)
					temp[j] = s[j];
				//compare temp to r
				double tempCost = costUpdate(temp, numVer);
				if (tempCost < cr) {
					cr = tempCost;
					for (int k = 0; k <= numVer; ++k)
						r[k] = temp[k];
				}
			}

			//threshold = 0
			int th = 0;
			if (cr - cs < th) {
				for (int i = 0; i <= numVer; ++i)
					s[i] = r[i];
				cs = cr;
			}

			//ending k = n ^ 2 EXITING CRITERIA HERE!!!!!!!!!!!!!!!!!!!!!!!
			k++;
		} while (k < numVer * numVer);

	}//end trial

	cout << globalCost << endl;
	for (int i = 1; i <= numVer; ++i)
		cout << globalOrder[i] << " ";
	cout << endl;

	//fclose
	fclose(stdin);

	return 0;
}