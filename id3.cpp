/* Author:	Ryan Armstrong
 * Project:	OLA 3 
 * Due:		November 15, 2018
 * Instructor: 	Dr. Phillips
 * Course:	CSCI 4350-001
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

struct Dataset
{
	static int nFeats; 			// number of real-valued features in dataset
	static int nClasses;			// number of unique classifications
	static set<int>* classifications;	// unique classifications

	vector<double> features;
	int classification;

	// parse features and classification from string
	Dataset(string toParse)
	{
		classification = stoi(toParse.substr(toParse.rfind(" ")));
		string featstr = toParse.substr(0, toParse.rfind(" ") + 1);
		string feat = "";
		for (int i = 0; featstr[i] != '\0'; i++)
		{
			if (featstr[i] != ' ')
			{
				feat += featstr[i];
			}
			else
			{
				features.push_back(stod(feat));
				feat = "";
			}
		}
	}

	// print the dataset
	void print()
	{
		for (double val : features)
			cout << val << " ";
		cout << classification << endl;
	}
};
int Dataset::nFeats = 0; 	// should always be > 0
int Dataset::nClasses = 0; 	// should always be > =
set<int>* Dataset::classifications = NULL;

struct Node
{
	Node* parent;		// parent
	Node* lChild;		// left child
	Node* rChild;		// right child
	int classification;	// classification of the node if terminal node. -1 otherwise.
	int attr;		// treating input as a list of vectors, the index of the attribute we've split on. -1 if terminal node.
	double cutoff;		// value of the cutoff for each split made. NaN if terminal node.
	bool terminal;		// is the node a terminal node?

	Node(Node* p) : parent(p), lChild(NULL), rChild(NULL), classification(-1), attr(-1), cutoff(NAN), terminal(false) {}
	Node(Node* p, int c) : parent(p), lChild(NULL), rChild(NULL), classification(c), attr(-1), cutoff(NAN), terminal(true) {}
	Node(Node* p, int a, double cut) : parent(p), lChild(NULL), rChild(NULL), classification(-1), attr(a), cutoff(cut), terminal(false) {}

	~Node()
	{
		if (lChild != NULL) delete lChild;
		if (rChild != NULL) delete rChild;
	}

	void print()
	{
		cout << "Parent: " << parent << endl;
		cout << "lChild: " << lChild << endl;
		cout << "rChild: " << rChild << endl;
		
		if (terminal)
		{
			cout << "Class: " << classification << endl;
		}
		else
		{
			cout << "Attribute: " << attr << endl;
			cout << "Cutoff: " << cutoff << endl;
		}
	}
};

struct DecisionTree
{
	Node* head;

	DecisionTree(vector<Dataset*> trainingData)
	{
		head = new Node(NULL);
		buildTree(head, trainingData);
	}

	~DecisionTree()
	{
		delete head;
	}

	// build a subtree
	void buildTree(Node* root, vector<Dataset*> data)
	{
		#ifdef DBG_BLDTREE
		cout << "Size of data vector passed to buildTree: " << data.size() << endl;
		for (Dataset* ds : data)
			ds->print();
		cout << endl;
		#endif

		// find the attribute with the largest gain
		vector<pair<double, double>> attrGains; // stores the best gain and the cutoff that produced it for each attribute
		for (int attr = 0; attr < Dataset::nFeats; attr++)
		{
			// sort data by ascending attribute value
			sort
			(
				data.begin(), 
				data.end(),
				[attr](const Dataset* a, const Dataset* b)
				{
					return a->features[attr] < b->features[attr];
				}
			);

			// get a vector of all cutoff values for potential splits in the attribute
			vector<double> splitCutoffs;
			vector<pair<double, double>> splitGains;
			for (int i = 0; i < data.size() - 1; i++)
				if (data[i]->features[attr] != data[i + 1]->features[attr])
					splitCutoffs.push_back((data[i]->features[attr] + data[i + 1]->features[attr]) / 2.0);
			
			#ifdef DBG_BLDTREE
			cout << "Size of splitCutoffs vector: " << splitCutoffs.size() << endl;
			for (double cutoff : splitCutoffs)
				cout << cutoff << " ";
			cout << endl;
			cout << endl;
			#endif

			// no potential splits found for attribute
			if (splitCutoffs.size() == 0)
			{
				attrGains.push_back(make_pair(0.0, NAN));
				continue;
			}

			// calculate gains for all potential splits
			for (double cutoff : splitCutoffs)
				splitGains.push_back(make_pair(calculateAttrSplitGain(attr, cutoff, data), cutoff));
			
			#ifdef DBG_BLDTREE
			cout << "Size of splitGains vector: " << splitGains.size() << endl;
			for (pair<double, double> gain : splitGains)
				cout << gain.first << "\t" << gain.second << endl;
			cout << endl;
			#endif

			// find cutoff split value with largest gain
			pair<double, double> bestGain = splitGains[0];
			for (pair<double, double> gainCutoffPair : splitGains)
				if (!isnan(gainCutoffPair.second) && gainCutoffPair.first > bestGain.first)
					bestGain = gainCutoffPair;

			attrGains.push_back(bestGain);
		}
		
		#ifdef DBG_BLDTREE
		cout << "attrGains size: " << attrGains.size() << endl;
		cout << endl;
		#endif

		// check to see if we've found any potential splits among the attributes
		// if not, terminal node reached
		if (attrGains.size() == 0)
		{
			// use majority class label, favoring smaller integer label
			map<int, int> classCounts;
			for (int c : *(Dataset::classifications))
				classCounts[c] = 0;
				
			for (Dataset* ds : data)
				classCounts.find(ds->classification)->second++;

			int classification = classCounts.begin()->first;
			for (auto it = classCounts.begin(); it != classCounts.end(); it++)
				classification = (it->second > classCounts[classification]) ? it->first : classification;

			Node* newRoot = new Node(root->parent, classification);
			delete root;
			root = newRoot;
			
			if (root->parent->lChild == NULL)
				root->parent->lChild = root;
			else
				root->parent->rChild = root;
		}

		pair<double, double> bestGain = attrGains[0];
		int bestAttr = 0;
		for (int i = 0; i < attrGains.size(); i++)
		{
			#ifdef DBG_BLDTREE
			cout << "Attribute " << i << " gain: " << attrGains[i].first << endl;
			#endif

			if (attrGains[i].first > bestGain.first)
			{
				bestGain = attrGains[i];
				bestAttr = i;
			}
		}

		// set up root as split node
		root->attr = bestAttr;
		root->cutoff = bestGain.second;

		#ifdef DBG_BLDTREE
		cout << endl;
		cout << "Best attribute: " << bestAttr << endl;
		cout << "Gain: " << bestGain.first << "; Cutoff: " << bestGain.second << endl;
		cout << endl;
		#endif

		// sort by ascending value of best attribute
		sort
		(
			data.begin(),
			data.end(),
			[bestAttr](const Dataset* a, const Dataset* b)
			{
				return a->features[bestAttr] < b->features[bestAttr];
			}
		);

		#ifdef DBG_BLDTREE
		cout << "Datasets sorted by best attribute, ascending: " << endl;
		for (Dataset* ds : data)
			ds->print();
		cout << endl;
		#endif

		// Split the data vector into two separate vectors based on the calculated best cutoff
		vector<Dataset*> LTCutoffData;
		vector<Dataset*> GTECutoffData;
		for (Dataset* ds : data)
		{
			#ifdef DBG_BLDTREE
			cout << "Dataset attr " << bestAttr << " value: " << ds->features[bestAttr] << endl;
			#endif

			if (ds->features[bestAttr] < bestGain.second)
				LTCutoffData.push_back(ds);
			else
				GTECutoffData.push_back(ds);
		}

		#ifdef DBG_BLDTREE
		cout << endl;
		cout << "LTCutoffData size: " << LTCutoffData.size() << endl;
		cout << "GTECutoffData size: " << GTECutoffData.size() << endl;
		cout << endl;
		#endif
		
		// scan through each new vector to see if we've made a terminal node
		// if not terminal node, continue building tree recursively
		bool leftTerminal = true;
		for (int i = 1; i < LTCutoffData.size(); i++)
		{
			if (LTCutoffData[i - 1]->classification != LTCutoffData[i]->classification)
			{
				leftTerminal = false;
				break;
			}
		}

		if (LTCutoffData.size() > 0)
		{
			if (leftTerminal)
			{
				#ifdef DBG_BLDTREE
				cout << "Terminal node reached on left child, classification: " << LTCutoffData[0]->classification << endl;
				cout << endl;
				#endif

				root->lChild = new Node(root, LTCutoffData[0]->classification);
			}
			else
			{
				root->lChild = new Node(root, bestAttr, bestGain.second);
				buildTree(root->lChild, LTCutoffData);
			}
		}

		bool rightTerminal = true;
		for (int i = 1; i < GTECutoffData.size(); i++)
		{
			if (GTECutoffData[i - 1]->classification != GTECutoffData[i]->classification)
			{
				rightTerminal = false;
				break;
			}
		}

		if (GTECutoffData.size() > 0)
		{
			if (rightTerminal)
			{
				#ifdef DBG_BLDTREE
				cout << "Terminal node reached on right child, classification: " << GTECutoffData[0]->classification << endl;
				cout << endl;
				#endif

				root->rChild = new Node(root, GTECutoffData[0]->classification);
			}
			else
			{
				root->rChild = new Node(root, bestAttr, bestGain.second);
				buildTree(root->rChild, GTECutoffData);
			}
		}
	}

	double calculateAttrSplitGain(int attr, double cutoff, vector<Dataset*> data)
	{
		#ifdef DBG_GAIN
		cout << "**********" << endl;
		cout << "ATTRIBUTE " << attr << "; CUTOFF " << cutoff << endl;
		cout << endl;
		#endif

		double gain;
		
		vector<double> probLTCutoff;	// probability of classifications given attr less than cutoff
		vector<double> probGTECutoff;	// probability of classifications given attr greater than or equal to cutoff

		// get total number of datasets whose chosen attribute is below and at/above cutoff
		int totalLTCount = 0;
		int totalGTECount = 0;
		for (Dataset* ds : data)
		{
			if (ds->features[attr] < cutoff)
				totalLTCount++;
			else
				totalGTECount++;
		}

		#ifdef DBG_GAIN
		cout << "# datasets with attrib val < " << cutoff << ": " << totalLTCount << endl;
		cout << "# datasets with attrib val >= " << cutoff << ": " << totalGTECount << endl;
		cout << endl;
		#endif

		// calculate P(classification | < cutoff) and P(classification | >= cutoff)
		for (int classification = 0; classification < Dataset::nClasses; classification++)
		{
			int LTCount = 0;
			int GTECount = 0;
			for (int i = 0; i < data.size(); i++)
			{
				if (data[i]->classification == classification && data[i]->features[attr] < cutoff)
					LTCount++;
				else if (data[i]->classification == classification && data[i]->features[attr] >= cutoff)
					GTECount++;
			}

			#ifdef DBG_GAIN
			cout << "# classifications " << classification << " given attrib val < " << cutoff << ": " << LTCount << endl;
			cout << "P(" << classification << " | < " << cutoff << ") = " << LTCount / (double)totalLTCount << endl;
			cout << "# classifications " << classification << " given attrib val >= " << cutoff << ": " << GTECount << endl;
			cout << "P(" << classification << " | >= " << cutoff << ") = " << GTECount / (double)totalGTECount << endl;
			cout << endl;
			#endif
			
			probLTCutoff.push_back(LTCount / (double)totalLTCount);
			probGTECutoff.push_back(GTECount / (double)totalGTECount);
		}

		// calculate I(classes | < cutoff) and I(classes | >= cutoff)
		double infoLTCutoff = 0.0;
		double infoGTECutoff = 0.0;
		for (int i = 0; i < Dataset::nClasses; i++)
		{
			infoLTCutoff += (probLTCutoff[i] != 0) ? -probLTCutoff[i] * log2(probLTCutoff[i]) : 0;
			infoGTECutoff += (probGTECutoff[i] != 0) ? -probGTECutoff[i] * log2(probGTECutoff[i]) : 0;
		}

		#ifdef DBG_GAIN
		cout << "I(classes | < " << cutoff << ") = " << infoLTCutoff << endl;
		cout << "I(classes | >= " << cutoff << ") = " << infoGTECutoff << endl;
		cout << endl;
		#endif

		// calculate E(attr ~ cutoff)
		double expected = (totalLTCount / (double)data.size()) * infoLTCutoff + (totalGTECount / (double)data.size()) * infoGTECutoff;

		#ifdef DBG_GAIN
		cout << "E(" << attr << " ~ " << cutoff << ") = " << expected << endl;
		cout << endl;
		#endif

		// calculate P(classification)
		map<int, int> classCounts; // number of entries for each classification
		map<int, double> pVals; // probability of each classification
		map<int, double> hVals; // shannon entropy of each classification

		// initialize count of each class to 0
		for (int c : *(Dataset::classifications))
			classCounts[c] = 0;

		for (Dataset* ds : data)
			classCounts.find(ds->classification)->second++;

		for (auto it = classCounts.begin(); it != classCounts.end(); it++)
		{
			pVals[it->first] = it->second / (double)data.size();
			hVals[it->first] = (pVals[it->first] != 0) ? -pVals[it->first] * log2(pVals[it->first]) : 0;
		}

		#ifdef DBG_GAIN
		for (auto it = classCounts.begin(); it != classCounts.end(); it++)
		{
			int c = it->first;
			cout << "num " << c << ": " << it->second << "; ";
			cout << "P(" << c << ") = " << pVals[c] << "; ";
			cout << "H(" << c << ") = " << hVals[c] << endl;
		}
		cout << endl;
		#endif

		// calculate I(classes)
		double info = 0;
		for (map<int, double>::iterator it = hVals.begin(); it != hVals.end(); it++)
			info += it->second;

		#ifdef DBG_GAIN
		cout << "I(classes) = " << info << endl;
		cout << endl;
		#endif

		// calculate gain
		gain = info - expected;

		#ifdef DBG_GAIN
		cout << "gain = " << gain << endl;
		cout << endl;
		#endif

		return gain;
	}

	// use the decision tree to deterimine the classification of something given a vector of feature
	// return the identified classification
	int identify(Dataset* ds)
	{
		Node* node = head;
		while (!node->terminal)
		{
			if (ds->features[node->attr] < node->cutoff)
			{
				#ifdef DBG_IDENT
				cout << "Attr " << node->attr << "; " << ds->features[node->attr] << " < " << node->cutoff << ", moving to left child." << endl;
				#endif

				node = node->lChild;
			}
			else
			{
				#ifdef DBG_IDENT
				cout << "Attr " << node->attr << "; " << ds->features[node->attr] << " >= " << node->cutoff << ", moving to right child." << endl;
				#endif

				node = node->rChild;
			}
		}

		#ifdef DBG_IDENT
		cout << "Terminal node reached.  Dataset Classification: " << ds->classification << "; Identified classification: " << node->classification << endl;
		cout << endl;
		#endif

		return node->classification;
	}
};

int main(int argc, char** argv)
{
	if (argc != 4)
	{
		cout << "Program takes three arguments." << endl;
		return -1;
	}
	
	Dataset::nFeats = atoi(argv[1]);

	ifstream training; training.open(argv[2]); 
	ifstream testing; testing.open(argv[3]);
	vector<Dataset*> trainingData;
	vector<Dataset*> testingData;
	string data;	// only used for reading in data from training and testing text files
	
	// read in training data
	getline(training, data);
	while (!training.eof())
	{
		trainingData.push_back(new Dataset(data));
		getline(training, data);
	}

	// set the number of classifications we're working with
	set<int> classifications;
	for (Dataset* ds : trainingData)
		if (classifications.insert(ds->classification).second)
			Dataset::nClasses++;
	Dataset::classifications = &classifications;

	// create decision tree
	DecisionTree dTree(trainingData);

	// read in testing data
	getline(testing, data);
	while (!testing.eof())
	{
		testingData.push_back(new Dataset(data));
		getline(testing, data);
	}

	// print number of testing example the algorithm correctly identifies
	int numCorrect = 0;
	for (Dataset* ds : testingData)
		if (dTree.identify(ds) == ds->classification)
			numCorrect++;
	cout << numCorrect << endl;

	training.close();
	testing.close();
	for (Dataset* d : trainingData)
		delete d;

	for (Dataset* d : testingData)
		delete d;

	return 0;
}
