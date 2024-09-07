package io.github.orionlibs.project_name;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.orionlibs.project_name.log.ListLogHandler;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;

@TestInstance(Lifecycle.PER_CLASS)
//@Execution(ExecutionMode.CONCURRENT)
public class NewClassTest
{
    private ListLogHandler listLogHandler;


    @BeforeEach
    void setUp()
    {
        listLogHandler = new ListLogHandler();
        NewClass.addLogHandler(listLogHandler);
    }


    @AfterEach
    public void teardown()
    {
        NewClass.removeLogHandler(listLogHandler);
    }


    @Test
    void test_method1() throws Exception
    {
        assertTrue(true);
        /*ConfigurationService.updateProp("orionlibs.prop", "false");
        mockMvc.perform(get("/")).andExpect(status().isOk());
        assertTrue(listLogHandler.getLogRecords().stream()
                        .anyMatch(record -> record.getMessage().contains("hello world")));
        ConfigurationService.updateProp("orionlibs.prop", "true");*/
    }
}
