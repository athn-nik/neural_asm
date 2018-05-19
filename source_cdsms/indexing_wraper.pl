#!/usr/bin/perl

$ws = $ARGV[0]; chomp($ws); ## Context window size
$voc_file = $ARGV[1]; chomp($voc_file); ## Path to vocabulary that defines both the rows and columns
                                        ## of the word-feature matrix
$corpus_file = $ARGV[2]; chomp($corpus_file); ## Path to corpus


$outd = "dsm/"; if (!(-e $outd)) { mkdir $outd; }
$prefix_for_outputs = $outd."index.ws".$ws;

## Indexing
##---------
system ("./indexing_core $voc_file $corpus_file $prefix_for_outputs $ws");
