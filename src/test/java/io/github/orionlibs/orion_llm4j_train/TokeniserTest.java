package io.github.orionlibs.orion_llm4j_train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;

@TestInstance(Lifecycle.PER_CLASS)
//@Execution(ExecutionMode.CONCURRENT)
public class TokeniserTest extends ATest
{
    private static final List<String> testStrings = List.of("", "?", "hello world!!!? (안녕하세요!) lol123 😉");
    private static final String specialsString = """
                    <|endoftext|>Hello world this is one document
                    <|endoftext|>And this is another document
                    <|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
                    <|endoftext|>Last document!!! 👋<|endofprompt|>
                    """.strip();
    private static final Map<String, Integer> specialTokens = Map.of(
                    "<|endoftext|>", 100257,
                    "<|fim_prefix|>", 100258,
                    "<|fim_middle|>", 100259,
                    "<|fim_suffix|>", 100260,
                    "<|endofprompt|>", 100276
    );
    private static final String llamaText = """
                    <|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
                    Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
                    The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
                    <|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
                    """;


    private String unpackText(String text)
    {
        if(text.startsWith("file:"))
        {
            return loadTestResource(text.substring("file:".length()));
        }
        else
        {
            return text;
        }
    }


    @Test
    void test_encodeDecodeIdentity_using_BytePairEncodingTokeniser()
    {
        BytePairEncodingTokeniser tokeniser = new BytePairEncodingTokeniser();
        for(int i = 0; i < testStrings.size(); i++)
        {
            List<Integer> encoding = tokeniser.encode(unpackText(testStrings.get(i)));
            String decoded = tokeniser.decode(encoding);
            assertEquals(testStrings.get(i), decoded);
        }
        String taylorSwiftText = unpackText("file:/io/github/orionlibs/orion_llm4j_train/taylorswift.txt");
        List<Integer> encoding = tokeniser.encode(taylorSwiftText);
        String decoded = tokeniser.decode(encoding);
        assertEquals(taylorSwiftText, decoded);
    }


    @Test
    void test_encodeDecodeIdentity_using_RegExTokeniser()
    {
        RegExTokeniser tokeniser = new RegExTokeniser(null);
        for(int i = 0; i < testStrings.size(); i++)
        {
            List<Integer> encoding = tokeniser.encode(unpackText(testStrings.get(i)));
            String decoded = tokeniser.decode(encoding);
            assertEquals(testStrings.get(i), decoded);
        }
        String taylorSwiftText = unpackText("file:/io/github/orionlibs/orion_llm4j_train/taylorswift.txt");
        List<Integer> encoding = tokeniser.encode(taylorSwiftText);
        String decoded = tokeniser.decode(encoding);
        assertEquals(taylorSwiftText, decoded);
    }


    @Test
    void test_handlingOfSpecialTokens()
    {
        RegExTokeniser tokeniser = new RegExTokeniser(null);
        List<Integer> encoding = tokeniser.encode(specialsString, AllowedSpecialTokenMode.ALL);
        String decoded = tokeniser.decode(encoding);
        assertEquals(specialsString, decoded);
    }


    @Test
    void test_handlingOfSpecialTokens_using_BytePairEncodingTokeniser()
    {
        BytePairEncodingTokeniser tokeniser = new BytePairEncodingTokeniser();
        List<Integer> encoding = tokeniser.encode(specialsString);
        String decoded = tokeniser.decode(encoding);
        assertEquals(specialsString, decoded);
    }


    @Test
    void test_savingAndLoadingModel_using_RegExTokeniser_and_GPT2Pattern() throws IOException
    {
        RegExTokeniser tokeniser = new RegExTokeniser(RegExTokeniser.GPT2_SPLIT_PATTERN);
        tokeniser.train(llamaText, 256 + 2);
        tokeniser.registerSpecialTokens(new HashMap<>());
        List<Integer> tokenIDs = tokeniser.encode(llamaText, AllowedSpecialTokenMode.ALL);
        assertEquals(llamaText, tokeniser.decode(tokenIDs));
        String tmpDir = System.getProperty("java.io.tmpdir");
        tokeniser.saveModel(tmpDir + "/test_tokenizer_tmp");
        tokeniser = new RegExTokeniser(null);
        tokeniser.loadModel(tmpDir + "/test_tokenizer_tmp.model");
        assertEquals(llamaText, tokeniser.decode(tokenIDs));
        assertEquals(llamaText, tokeniser.decode(tokeniser.encode(llamaText, AllowedSpecialTokenMode.ALL)));
        assertEquals(tokenIDs, tokeniser.encode(llamaText, AllowedSpecialTokenMode.ALL));
        new File(tmpDir + "/test_tokenizer_tmp.model").delete();
        new File(tmpDir + "/test_tokenizer_tmp.vocab").delete();
    }


    @Test
    void test_savingAndLoadingModel_using_RegExTokeniser_and_GPT4Pattern() throws IOException
    {
        RegExTokeniser tokeniser = new RegExTokeniser(RegExTokeniser.GPT4_SPLIT_PATTERN);
        tokeniser.train(llamaText, 256 + 2);
        tokeniser.registerSpecialTokens(new HashMap<>());
        List<Integer> tokenIDs = tokeniser.encode(llamaText, AllowedSpecialTokenMode.ALL);
        assertEquals(llamaText, tokeniser.decode(tokenIDs));
        String tmpDir = System.getProperty("java.io.tmpdir");
        tokeniser.saveModel(tmpDir + "/test_tokenizer_tmp");
        tokeniser = new RegExTokeniser(null);
        tokeniser.loadModel(tmpDir + "/test_tokenizer_tmp.model");
        assertEquals(llamaText, tokeniser.decode(tokenIDs));
        assertEquals(llamaText, tokeniser.decode(tokeniser.encode(llamaText, AllowedSpecialTokenMode.ALL)));
        assertEquals(tokenIDs, tokeniser.encode(llamaText, AllowedSpecialTokenMode.ALL));
        new File(tmpDir + "/test_tokenizer_tmp.model").delete();
        new File(tmpDir + "/test_tokenizer_tmp.vocab").delete();
    }
}
