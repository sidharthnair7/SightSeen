package dev.sightseen.request;

import com.jayway.jsonpath.JsonPath;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * The request flow without a GPU. The 3D helper is pointed at a port where nothing listens, so the test also checks
 * that the student is told why it stopped.
 */
@SpringBootTest(properties = {
        "sightseen.data-dir=target/test-data",
        "spring.datasource.url=jdbc:h2:mem:sightseen-test",
        "sightseen.worker.url=http://127.0.0.1:9"
})
@AutoConfigureMockMvc
class RequestFlowTest {

    @Autowired
    MockMvc mvc;

    private String create(String label) throws Exception {
        return mvc.perform(post("/api/requests").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"label\":\"" + label + "\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("WAITING_FOR_VIDEO"))
                .andReturn().getResponse().getContentAsString();
    }

    @Test
    void landlordLinkTakesTheVideoAndTheStudentSeesWhyItStopped() throws Exception {
        String created = create("Room on Champlain Cres");
        String id = JsonPath.read(created, "$.id");
        String landlordPath = JsonPath.read(created, "$.landlordPath");
        String token = landlordPath.substring("/film/".length());

        // The landlord sees which place it's for, and nothing of the student's
        mvc.perform(get("/api/film/" + token))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.label").value("Room on Champlain Cres"))
                .andExpect(jsonPath("$.videoReceived").value(false))
                .andExpect(jsonPath("$.id").doesNotExist());

        MockMultipartFile video = new MockMultipartFile("video", "room.mp4", "video/mp4", new byte[]{0, 0, 0, 24});
        mvc.perform(multipart("/api/film/" + token + "/video").file(video))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.videoReceived").value(true));

        mvc.perform(get("/api/requests/" + id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("FAILED"))
                .andExpect(jsonPath("$.videoSource").value("LANDLORD_LINK"))
                .andExpect(jsonPath("$.failureMessage").isNotEmpty());
    }

    @Test
    void aFileThatIsNotAVideoIsTurnedAway() throws Exception {
        String id = JsonPath.read(create("Basement on George St"), "$.id");
        MockMultipartFile notes = new MockMultipartFile("video", "notes.txt", "text/plain", "hi".getBytes());

        mvc.perform(multipart("/api/requests/" + id + "/video").file(notes))
                .andExpect(status().isBadRequest());
    }

    @Test
    void unknownLinksDontWork() throws Exception {
        mvc.perform(get("/api/film/not-a-real-token")).andExpect(status().isNotFound());
        mvc.perform(get("/api/requests/not-a-real-id")).andExpect(status().isNotFound());
    }
}
