/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*- */
/*
 * main.cc
 * Copyright (C) Elias 2013 <iosife@Elias-HP-ProBook-4530s>
 * 
CorpusProcessing is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * CorpusProcessing is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * This a customized indexing program.
 */

#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>
#include <cerrno>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include </usr/local/include/sparsehash/sparse_hash_map>

using namespace std;
using google::sparse_hash_map;
using std::tr1::hash;


// The For loop way
void toLowerCase(std::string &str)
{
	const int length = str.length();
	for(int i=0; i < length; ++i)
	{
		str[i] = std::tolower(str[i]);
	}
}

int main(int argc, char* argv[])
{
	// For corpus
	//int CORPUS_SIZE = 116755354;
   // double progress = 0; // Progress
	unsigned long int num_lines = 0; // Number of corpus lines
	unsigned int pos = 0; //***// Position of every token in a sentence
	string corpus_line; // Current corpus line
    string sbuf; // Buffer for string
	string cur_token; // Current token of line
	stringstream ss(corpus_line); // String stream
    vector<string> line_tokens; // Vector holding line tokens
    vector<unsigned int> line_token_positions; //***// Vector holding the positions of every token f the current line;
    vector<unsigned int>::iterator it_pos1; //***// Iterator of the above vector
    vector<unsigned int>::iterator it_pos2; //***// Iterator of the above vector
	vector<string>::iterator it1; // Iterator of vector of strings
	// For word frequency
    sparse_hash_map<unsigned long int, unsigned long int> wfreq;
	unsigned long int token_idx; // Index of token wrt dictionary
    //unsigned long int  lsize = 0; // Size of hash
	// For dictionary
	unsigned long int dict_idx_counter = 0; // Index for dictionary entries
    sparse_hash_map<string, int> dct; // Dictionary as hash
    string dct_line; // Current dictionary line
    sparse_hash_map<string, int>::iterator it2; // Iterator for dictionary
	string cur_dict_entry_word; // Current entry of dictionary: word (key)
	//unsigned long int cur_dict_entry_idx; // Current entry of dictionary: index (value)
    unsigned long int  dct_size = 0;
    // General statistics
	unsigned long int  total_num_tokens = 0;
	// Main structs and related iterators
	vector<unsigned long int> line_tok_in_dct_idx; // Indices of tokens of current line, which are included in dictionary
    vector<unsigned long int>::iterator it31; // Iterator for the above vector
	vector<unsigned long int>::iterator it32; // Iterator for the above vector. Related with it31. Both in double loop.
	sparse_hash_map<unsigned long int, sparse_hash_map<unsigned long int, unsigned long int> > ccfreq;
    sparse_hash_map<unsigned long int, unsigned long int> cur_cwf_value; // Value of ccfreq
	sparse_hash_map<unsigned long int, unsigned long int>::iterator it4; // Iterator for the value of ccfreq 
    // Output
	stringstream out_prefix_ss1;
	stringstream out_prefix_ss2;
    string out_prefix_s1;
	string out_prefix_s2;
	string freq_f_extension = ".freq";
	string index_f_extension = ".index";
	string out_freq_file;
	string out_index_file;
	int context_window;
	int  word_distance;
	// How to run the program
    if (argc != 5) 
	 {
	  cerr << "Wrong number of arguments." << endl;
      cerr << "Usage: " << argv[0] << " dictionary corpus prefix_for_output_files corpus context_window[integer > 0]" << endl;         
      return 1;
     }

	
	// Covert input arguments into appropriate format
    out_prefix_ss1 << argv[3];
    out_prefix_ss1 >> out_prefix_s1;
	out_prefix_ss2 << argv[3];
	out_prefix_ss2 >> out_prefix_s2;
	// and create the names of output files
	out_freq_file = out_prefix_s1.append(freq_f_extension);
	out_index_file = out_prefix_s2.append(index_f_extension);
	
	//assign the context_window arg to a variable
	context_window = atoi(argv[4]);
	// Few messages
	cout << endl;
	cout << "CORPUS INDEXING" << endl;
	cout << "*** Based on the co-occurrence of words included in the given dictionary. ***" << endl;
	cout << "*** Word co-occurrence is considered at (corpus) sentence level.  ***" << endl;
	cout << "*** Only words that cooccur inside the context window are considered. ***" << endl << endl;
	cout << "The following files will be created: " << endl;
	cout << " [1] " << out_freq_file << ": Word occurrence frequencies." << endl;
	cout << " [2] " << out_index_file << ": Word co-occurrence frequencies. Only non-zero shown, i.e., non-sparse representation." << endl << endl;
	cout << "* Indexing: started" << endl;

	
	//------------------------------------------------------------------------//
    // Load dictionary                                                        //
	//------------------------------------------------------------------------//
	cout << "Loading reference dictionary: started." << endl;
	ifstream dictionary(argv[1]);
	
    //------------------------------------------------------------------------//
	// Check file for reading                                                 //
	//------------------------------------------------------------------------//
    if (!dictionary.is_open())
	 {		
      cerr << "Error while opening dictionary file" << endl;
	  return 1;
	 }
	if (dictionary.bad())
	{
     cerr << "Error while reading dictionary file" << endl;
	 return 1;
	}

    //------------------------------------------------------------------------//
    // Read dictiοnary line by line                                            //
	//------------------------------------------------------------------------//
    while(getline(dictionary, dct_line, '\n'))
	 {
	   dict_idx_counter++;
	   dct[dct_line] = dict_idx_counter; // Update dictionary
	   //cout  << " Dictionary entry-->" << dct_line << "<---Index--->" << dict_idx_counter << "<---" << endl;
	 }
	dictionary.close(); // Close dictionary file
	cout << "Loading reference dictionary: completed." << endl;

	//------------------------------------------------------------------------//
	// Just traverse the loaded dictionary                                    //
	// For debugging only. How to:                                            //
	// 1. Iterate over hash                                                   //
	// 2. Access (retrieve) keys & values                                     //
	//------------------------------------------------------------------------//
	//for(it2=dct.begin(); it2 != dct.end(); it2++)
	 //{
	    //cur_dict_entry_word = it2->first; // Key: word
		//cur_dict_entry_idx = it2->second; // Value: index
		//cout << "Dictionary entry--->" << cur_dict_entry_word << "<---" << "Index-->" << cur_dict_entry_idx << "<---" << endl;
	 //}
	
	
	ifstream corpus(argv[2]); // Corpus file
	//------------------------------------------------------------------------//
	// Check file for reading                                                 //
	//------------------------------------------------------------------------//
    if (!corpus.is_open())
	 {		
      cerr << "Error while opening corpus file" << endl;
	  return 1;
	 }
	if (corpus.bad())
	{
     cerr << "Error while reading corpus file" << endl;
	 return 1;
	}

	//------------------------------------------------------------------------//
    // Read corpus line by line                                               //
	//------------------------------------------------------------------------//
    while(getline(corpus, corpus_line, '\n'))
	 {
		 
      num_lines ++;
	  //progress = (double)num_lines/CORPUS_SIZE;
	  //lsize = wfreq.size();
	//  cout  << "Progress: " << progress << " Size: " << lsize << endl; // Prints progress and lexicon size
          cout  << num_lines << endl;

	  stringstream ss(corpus_line); // reading the corpus line
	  line_tokens.clear(); // Clear (i.e., initialize) vector 
      while (ss >> sbuf) line_tokens.push_back(sbuf); // Tokenize line
		 
	  // Traverse tokens of current line
	  line_tok_in_dct_idx.clear(); // Clear (i.e., initialize) vector for each line
	  line_token_positions.clear();
	  for(it1=line_tokens.begin(); it1 < line_tokens.end(); it1++)
	   {  
		  total_num_tokens++; // Number of tokens in corpus 
		  pos ++;	//position of current token in sentence;
		  cur_token = *it1; // Convert iterator to string: current token
      	//  cout  <<  "Current Token : " << cur_token << endl; // Prints current token
		  
		  toLowerCase(cur_token); 
		  //cout << "Lowercased token -->" << cur_token << endl;
		  //---------------------------//
		  // If token is in dictionary //
		  //---------------------------//
		  if (dct.count(cur_token))
		   {
			 //  cout  << "** Exist ** " << endl;
			 token_idx = dct[cur_token]; // Get the index of current token (wrt dictionary)
			 line_tok_in_dct_idx.push_back (token_idx);
			 line_token_positions.push_back(pos);
			 
			 //cout  << " * Matched line" << num_lines << ": " << corpus_line << endl; // Prints current line  
			 //cout  << " * Matched line" << num_lines << ": token: " << cur_token << " Index: " << token_idx << endl;  
		     wfreq[token_idx] += 1; // Update hash of token frequencies: the value is the index of token
		   }
       } // For each token

	  it_pos1=line_token_positions.begin();
	  // Get the pairs of co-occurring tokens (their indices) in current line
      for(it31=line_tok_in_dct_idx.begin(); it31!=line_tok_in_dct_idx.end(); it31++)
	   {
		  //cout << " - Co-occ. indices: " << *it31 << endl;
		  it_pos2=line_token_positions.begin();
		  for(it32=line_tok_in_dct_idx.begin(); it32!=line_tok_in_dct_idx.end(); it32++)
		   {	
			  word_distance= *it_pos1 - *it_pos2;
			//  cout << " ** Word distance between: " << *it31 << " and " << *it32 << " is --> " << abs(word_distance) << endl;
			
			  if (*it31 != *it32)
			   {
				   if (abs(word_distance) <= context_window && word_distance !=0 ){
					 //  cout << " --Mpikap-- " <<endl;
					ccfreq[*it31][*it32] += 1;
					}
			   }
			  else
			   {
			    ccfreq[*it31][*it31] = wfreq[*it31]; // Co-occurs with itself 
			   }
			  //cout << " -- Co-occ. indices: " << *it32 << endl;
		   it_pos2++;
		   }
		   it_pos1++;
	   } // Get the pairs of co-occurring tokens (their indices) in current line
		 
		 pos=0;
     } // For each corpus line
     corpus.close(); // Close corpus file

	
	//------------------------------------------------------------------------//
	// Report some basic statistics                                           //
	//------------------------------------------------------------------------//
	cout << "* Indexing: completed" << endl << endl;
	cout << "* Basic Statistics *" << endl;
	dct_size = dct.size();
	cout  << " Number of dictionary entries: " << dct_size << endl;
	//lsize = wfreq.size();
	//cout << " Number of (unique) dictionary entries found in corpus: " << lsize << endl;
        cout << " Number of corpus lines: " << num_lines << endl;
	cout << " Number of corpus tokens (not unique): " << total_num_tokens << endl;
    

	
    // Open first output file
	ofstream out1;
	const char *out_freq_file_cstr = out_freq_file.c_str();
    out1.open(out_freq_file_cstr, ios::out); 
	if (out1.is_open())
    {
	//------------------------------------------------------------------------//
	// Report word frequencies for the words of dictionary.                   //
	// If a dictionary entry is not in wfreq, then the value is 0:            //
	// so, we don't need to introduce related checks.                         //
	//------------------------------------------------------------------------//
	for (unsigned long int wf_i=1; wf_i<(dct_size+1); wf_i++) // For each entry (its index) of dictionary
	 {
		out1 << wfreq[wf_i] << endl;
	 } // For each index of dictionary
	 out1.close(); // Close first output file
	 } // If out file can be opened for writing
	 else
	 {
	   cerr << "Can not write first output file " << out_freq_file << endl;
	   return 1;
	 }
     
	

	// Open second output file
	ofstream out2;
	const char *out_index_file_cstr = out_index_file.c_str();
    out2.open (out_index_file_cstr, ios::out);
	if (out2.is_open())
    {	
    //------------------------------------------------------------------------//
	// Report co-occurring tokens and respective frequency                    //
	//------------------------------------------------------------------------//
	for (unsigned long int cwf_i=1; cwf_i<(dct_size+1); cwf_i++) // For each entry (its index) of dictionary
	 {
		 if (ccfreq.count(cwf_i)) // The corresponding word co-occurs with some words of dictionary
		  {
			 cur_cwf_value = ccfreq[cwf_i];
			 for(it4=cur_cwf_value.begin(); it4 != cur_cwf_value.end(); it4++)
			  { 
			   out2 << it4->first << "," << it4->second << " ";
			  }
			 out2 << "\n";
		  }
		 else // The corresponding word do not co-occur with some words of dictionary
		  {
			 if (wfreq.count(cwf_i)) // Simply, output its own frequency
			  {
			   out2 << cwf_i << "," << wfreq[cwf_i] << " ";
			  }
			 else // Worst case (why? incomplete data acquisition, etc): output zero frequency
			  {
			   out2 << cwf_i << ",0";
			  }
			 printf ("\n");
			 out2 << "\n";
		  }
	 } // Report co-occurring tokens and respective frequency
	 out2.close(); // Close second output file
	} // If out file can be opened for writing
	else
	{
	 cerr << "Can not write second output file " << out_index_file << endl;
	 return 1;
	}


	
	return 0;
}

