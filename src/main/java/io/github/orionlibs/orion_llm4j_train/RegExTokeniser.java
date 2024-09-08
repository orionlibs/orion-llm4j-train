package io.github.orionlibs.orion_llm4j_train;

import io.github.orionlibs.orion_assert.Assert;
import io.github.orionlibs.orion_tuple.Pair;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * RegExTokeniser handles an optional RegEx splitting pattern.
 * RegExTokeniser handles optional special tokens.
 */
public class RegExTokeniser extends Tokeniser
{
    private static final String GPT2_SPLIT_PATTERN = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    private static final String GPT4_SPLIT_PATTERN = "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
    private String patternRegEx;
    private Pattern pattern;
    private Map<String, Integer> specialTokens;
    private Map<Integer, String> inverseSpecialTokens;


    public RegExTokeniser(String patternRegEx)
    {
        super();
        this.patternRegEx = patternRegEx != null && !patternRegEx.isEmpty() ? patternRegEx : GPT4_SPLIT_PATTERN;
        this.pattern = Pattern.compile(this.patternRegEx);
        this.specialTokens = new HashMap<>();
        this.inverseSpecialTokens = new HashMap<>();
    }


    @Override
    public void train(String text, int vocabularySize)
    {
        Assert.isGreaterOrEqualTo(vocabularySize, 256, "vocabularySize has to be >= 256");
        int numberOfMergesToDo = vocabularySize - 256;
        Matcher matcher = pattern.matcher(text);
        List<String> textChunks = new ArrayList<>();
        while(matcher.find())
        {
            textChunks.add(matcher.group());
        }
        // Input text preprocessing: convert chunks to lists of bytes casted to integers
        List<List<Integer>> tokenIDsForChunks = new ArrayList<>();
        for(String chunk : textChunks)
        {
            byte[] bytes = chunk.getBytes(StandardCharsets.UTF_8);
            List<Integer> byteList = new ArrayList<>();
            for(byte b : bytes)
            {
                byteList.add(b & 0xFF);
            }
            tokenIDsForChunks.add(byteList);
        }
        //iteratively merge the most common pairs to create new tokens
        Map<Pair<Integer, Integer>, Integer> mergesTemp = new HashMap<>();
        Map<Integer, byte[]> vocabularyTemp = new HashMap<>();
        for(int i = 0; i < 256; i++)
        {
            vocabularyTemp.put(i, new byte[] {(byte)i});
        }
        for(int i = 0; i < numberOfMergesToDo; i++)
        {
            Map<Pair<Integer, Integer>, Integer> frequenciesOfConsecutivePairs = new HashMap<>();
            for(List<Integer> tokenIDs : tokenIDsForChunks)
            {
                Utils.getFrequencyOfConsecutivePairs(tokenIDs, frequenciesOfConsecutivePairs);
            }
            //find the pair with the highest count
            Pair<Integer, Integer> pairWithHighestFrequency = frequenciesOfConsecutivePairs.entrySet()
                            .stream()
                            .max(Entry.comparingByValue())
                            .map(Entry::getKey)
                            .orElse(null);
            //mint a new token: assign it the next available id
            int idx = 256 + i;
            //replace all occurrences of pair in ids with idx
            List<List<Integer>> mergedTokenIDs = new ArrayList<>();
            for(List<Integer> tokenIDs : tokenIDsForChunks)
            {
                List<Integer> tokenIDsTemp = Utils.merge(tokenIDs, pairWithHighestFrequency, idx);
                mergedTokenIDs.add(tokenIDsTemp);
            }
            //save the merge
            mergesTemp.put(pairWithHighestFrequency, idx);
            byte[] firstArray = vocabularyTemp.get(pairWithHighestFrequency.getFirst());
            byte[] secondArray = vocabularyTemp.get(pairWithHighestFrequency.getSecond());
            byte[] temp = new byte[firstArray.length + secondArray.length];
            System.arraycopy(firstArray, 0, temp, 0, firstArray.length);
            System.arraycopy(secondArray, 0, temp, firstArray.length, secondArray.length);
            vocabularyTemp.put(idx, temp);
        }
        this.merges = mergesTemp;
        this.vocabulary = vocabularyTemp;
    }


    @Override
    public List<Integer> encode(String text)
    {
        return encode(text, AllowedSpecialTokenMode.NONE_RAISE);
    }


    public List<Integer> encode(String text, AllowedSpecialTokenMode allowedSpecialTokens)
    {
        Map<String, Integer> specialTokensToUse = null;
        if(allowedSpecialTokens == AllowedSpecialTokenMode.ALL)
        {
            specialTokensToUse = specialTokens;
        }
        else if(allowedSpecialTokens == AllowedSpecialTokenMode.NONE)
        {
            specialTokensToUse = new HashMap<>();
        }
        else if(allowedSpecialTokens == AllowedSpecialTokenMode.NONE_RAISE)
        {
            specialTokensToUse = new HashMap<>();
            boolean allTokensAbsent = specialTokens.keySet()
                            .stream()
                            .noneMatch(token -> text.contains(token));
            if(!allTokensAbsent)
            {
                throw new AssertionError("Not all special tokens are absent in the text.");
            }
        }
        if(specialTokensToUse.isEmpty())
        {
            return encodeOrdinaryText(text);
        }
        String specialPattern = "(" + String.join("|", specialTokens.keySet()) + ")";
        specialPattern = Pattern.quote(specialPattern);
        String[] specialChunks = text.split(specialPattern);
        //now all the special characters are separated from the rest of the text
        //all chunks of text are encoded separately, then results are joined
        List<Integer> tokenIDs = List.of();
        for(String chunk : specialChunks)
        {
            if(specialTokensToUse.containsKey(chunk))
            {
                //this is a special token, encode it separately as a special case
                tokenIDs.add(specialTokensToUse.get(chunk));
            }
            else
            {
                //this is an ordinary sequence, encode it normally
                tokenIDs.addAll(encodeOrdinaryText(chunk));
            }
        }
        return tokenIDs;
    }


    @Override
    public String decode(List<Integer> tokenIDs)
    {
        List<byte[]> partBytes = new ArrayList<>();
        for(Integer idx : tokenIDs)
        {
            if(vocabulary.containsKey(idx))
            {
                partBytes.add(vocabulary.get(idx));
            }
            else if(inverseSpecialTokens.containsKey(idx))
            {
                partBytes.add(inverseSpecialTokens.get(idx).getBytes(StandardCharsets.UTF_8));
            }
            else
            {
                throw new IllegalArgumentException("invalid token id: " + idx);
            }
        }
        // Combine all byte arrays into a single byte array
        int totalLength = partBytes.stream().mapToInt(b -> b.length).sum();
        byte[] textBytes = new byte[totalLength];
        int currentIndex = 0;
        for(byte[] byteArray : partBytes)
        {
            System.arraycopy(byteArray, 0, textBytes, currentIndex, byteArray.length);
            currentIndex += byteArray.length;
        }
        // Decode byte array to string
        return new String(textBytes, StandardCharsets.UTF_8);
    }


    public void registerSpecialTokens(Map<String, Integer> specialTokens)
    {
        this.specialTokens = specialTokens;
        this.inverseSpecialTokens = new HashMap<>();
        specialTokens.entrySet().forEach(e -> inverseSpecialTokens.put(e.getValue(), e.getKey()));
    }


    public List<Integer> encodeChunk(byte[] textBytes)
    {
        List<Integer> tokenIDs = new ArrayList<>();
        // Convert all bytes to integers in range 0..255
        for(byte textByte : textBytes)
        {
            tokenIDs.add(textByte & 0xFF);
        }
        while(tokenIDs.size() >= 2)
        {
            // Find the pair with the lowest merge index
            Map<Pair<Integer, Integer>, Integer> frequenciesOfConsecutivePairs = Utils.getFrequencyOfConsecutivePairs(tokenIDs);
            //find the pair with the lowest count
            Pair<Integer, Integer> pairWithLowestFrequency = frequenciesOfConsecutivePairs.entrySet()
                            .stream()
                            .min(Entry.comparingByValue())
                            .map(Entry::getKey)
                            .orElse(null);
            //subtle: if there are no more merges available, the key will
            //result in an inf for every single pair, and the min will be
            //just the first pair in the list, arbitrarily
            //we can detect this terminating case by a membership check
            if(!merges.containsKey(pairWithLowestFrequency))
            {
                break;//nothing else can be merged anymore
            }
            //otherwise let's merge the best pair (lowest merge index)
            Integer idx = merges.get(pairWithLowestFrequency);
            tokenIDs = Utils.merge(tokenIDs, pairWithLowestFrequency, idx);
        }
        return tokenIDs;
    }


    public List<Integer> encodeOrdinaryText(String text)
    {
        // Split text into chunks based on the compiled pattern
        Matcher matcher = pattern.matcher(text);
        List<String> textChunks = new ArrayList<>();
        while(matcher.find())
        {
            textChunks.add(matcher.group());
        }
        // Encode each chunk separately and then join the results
        List<Integer> ids = new ArrayList<>();
        for(String chunk : textChunks)
        {
            byte[] chunkBytes = chunk.getBytes(StandardCharsets.UTF_8);
            List<Integer> chunkIds = encodeChunk(chunkBytes);
            boolean addChunk = true;
            /*for(int chunkID : chunkIds)
            {
                if(chunkID < 0)
                {
                    addChunk = false;
                    break;
                }
            }*/
            if(addChunk)
            {
                ids.addAll(chunkIds);
            }
        }
        return ids;
    }
}
