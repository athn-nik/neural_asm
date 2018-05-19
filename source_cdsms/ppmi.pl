#!/usr/bin/perl

$ws = $ARGV[0]; chomp($ws);
$CORPUS_TOKENS = $ARGV[1]; chomp($CORPUS_TOKENS); ## Number of coprus tokens (not unique)
$VOCSIZE = $ARGV[2]; chomp($VOCSIZE); ## Number of vocabulary entries
$dsm_dimf = $ARGV[3]; chomp($dsm_dimf); ## Path to file that includes the indices of the words
                                        ## that constitute the columns of the word-features matrix.
                                        ## These words should be included in the vocabulary.
                                        ## Their index is identical to the vocabulary index.
$dim = $VOCSIZE;


## Files to load
##--------------
$indd = "dsm/";
$frf = $indd."index.ws".$ws.".freq";
$idf = $indd."index.ws".$ws.".index";

## Load vocab ids for DSM dimensions
##----------------------------------
%dsm = ();
$dsm_dim_counter = 0;
open (D,$dsm_dimf) || die "can not open $dsm_dimf\n";
$r = <D>;
while ($r ne "")
 {
  $dsm_dim_counter ++;
  chomp($r);
  $dsm{$r} = $dsm_dim_counter;
  $r = <D>;
 }
close (D);

## Load frequencies for entire vocabulary
##---------------------------------------
$c1 = 0;
%v2f = (); ## Key: word index, Value: corpus frequency
open (F,$frf) || die "can not open $frf\n";
$r = <F>;
while ($r ne "")
 {
   $c1++;
   chomp($r);
   $v2f{$c1} = $r + 1;
   $r = <F>;
 }
close (F);
print ("The frequencies for ",$c1," words loaded.\n");

##----------------------------------------##
## Create vector of zeros                 ##
##----------------------------------------##
@zerov = ();
for ($z=0; $z<=($dim-1); $z++) { $zerov[$z] = 0; }

##----------------------------------------##
## Work for each entry of the vocabulary: ##
## Load index                             ##
##----------------------------------------##
$of = $indd."index.ws".$ws.".ppmi";
open (O,">$of") || die "can not open $of for writing\n";
$nid = 0;
open (I,$idf) || die "can not open $idf\n";
$r = <I>;
while ($r ne "") ## For each line of index, i.e., for each vocabulary word
 {
     $ppmi_str = "";
     $non_empty_ppmi_str = 0;

     $nid ++; ## Index
     chomp($r);
     print ("Index line: ",$nid,"\n");
     @a1 = split(/\s+/,$r);
     foreach (@a1) ## For each co-occuring word
      {
        ($coocw,$coocf) = split(/,/,$_); ## word index, freq of co-occurrence
        if (defined($dsm{$coocw}))
         {
           if ($coocf <=  1) { $ppmi = 0; }
           else { $ppmi = log(($CORPUS_TOKENS*$coocf)/($v2f{$nid}*$v2f{$coocw})); }
           if ($ppmi < 0) { $ppmi = 0; }
           $vidx = $dsm{$coocw} - 1; ## Index for vector
           $ppmi = sprintf("%.3f", $ppmi);
           if ($ppmi > 0)
            {
              $ppmi_str .= $coocw.":".$ppmi." ";
              $non_empty_ppmi_str ++;
            }
         }
      } ## For each co-occuring word
     if ($non_empty_ppmi_str == 0) { $ppmi_str .= $nid.":1 ";}
     chop($ppmi_str);
     print O ($ppmi_str,"\n");
     $r = <I>;
 } ## For each line of index
close (I);
close (O);
